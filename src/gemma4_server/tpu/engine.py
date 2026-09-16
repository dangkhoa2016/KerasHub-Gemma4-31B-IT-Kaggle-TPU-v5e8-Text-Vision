from __future__ import annotations
import ctypes
import gc
import math, time
from pathlib import Path
from typing import Any

from ..core.model_path import weights_path
from .generation import (
    chat_prompt,
    vision_prompt,
    plan_authority_generation,
    plan_generation,
    scalar_text,
)
from .distribution import LAYOUT_PROFILE
from .sharded_checkpoint import sharded_checkpoint_assignment


def _discover_cgroup_memory_current_path():
    try:
        for line in Path("/proc/self/cgroup").read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                return Path("/sys/fs/cgroup") / parts[2].lstrip("/") / "memory.current"
    except OSError:
        pass
    return Path("/sys/fs/cgroup/memory.current")


_CGROUP_MEMORY_CURRENT_PATH = _discover_cgroup_memory_current_path()


def _read_rss_kib():
    try:
        for line in Path("/proc/self/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _read_cgroup_memory_current():
    try:
        return int(_CGROUP_MEMORY_CURRENT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def post_load_host_cleanup():
    """Release project-owned host allocations without touching model buffers."""
    before_rss = _read_rss_kib()
    before_cgroup = _read_cgroup_memory_current()
    collected = gc.collect()
    after_gc_rss = _read_rss_kib()
    after_gc_cgroup = _read_cgroup_memory_current()
    trim_available = False
    trim_result = None
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim")
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        trim_available = True
        trim_result = bool(malloc_trim(0))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    after_trim_rss = _read_rss_kib()
    after_trim_cgroup = _read_cgroup_memory_current()
    return {
        "post_load_rss_before_cleanup_kib": before_rss,
        "post_load_rss_after_gc_kib": after_gc_rss,
        "post_load_rss_after_malloc_trim_kib": after_trim_rss,
        "cgroup_memory_current_before_cleanup": before_cgroup,
        "cgroup_memory_current_after_gc": after_gc_cgroup,
        "cgroup_memory_current_after_malloc_trim": after_trim_cgroup,
        "post_load_gc_collected_objects": collected,
        "post_load_malloc_trim_available": trim_available,
        "post_load_malloc_trim_result": trim_result,
    }

def _dtype_itemsize(dtype):
    name = str(dtype).lower()
    for key, size in {
        "bool":1, "int8":1, "uint8":1,
        "float16":2, "bfloat16":2, "int16":2, "uint16":2,
        "float32":4, "int32":4, "uint32":4,
        "float64":8, "int64":8, "uint64":8,
    }.items():
        if key in name:
            return size
    return None

def _logical_nbytes(weight):
    value = getattr(weight, "value", weight)
    shape = getattr(value, "shape", getattr(weight, "shape", None))
    dtype = getattr(value, "dtype", getattr(weight, "dtype", None))
    if shape is None or dtype is None:
        return None
    itemsize = _dtype_itemsize(dtype)
    if itemsize is None:
        return None
    try:
        return math.prod(int(x) for x in shape) * itemsize
    except Exception:
        return None

def summarize_sharding(model, sample_limit=32):
    counts = {"sharded":0, "replicated":0, "unknown":0}
    bytes_ = {"sharded":0, "replicated":0, "unknown":0, "logical":0}
    samples = []
    for w in model.weights:
        value = getattr(w, "value", w)
        sharding = getattr(value, "sharding", None)
        replicated = (
            getattr(sharding, "is_fully_replicated", None)
            if sharding is not None else None
        )
        cls = (
            "replicated" if replicated is True
            else "sharded" if replicated is False
            else "unknown"
        )
        nbytes = _logical_nbytes(w)
        counts[cls] += 1
        if nbytes is not None:
            bytes_[cls] += nbytes
            bytes_["logical"] += nbytes
        if len(samples) < sample_limit:
            samples.append({
                "path": str(
                    getattr(w, "path", getattr(w, "name", "unknown"))
                ),
                "shape": [int(v) for v in getattr(w, "shape", ())],
                "sharding": (
                    str(sharding) if sharding is not None else None
                ),
                "fully_replicated": replicated,
                "logical_bytes": nbytes,
            })

    pct = (
        100.0 * bytes_["sharded"] / bytes_["logical"]
        if bytes_["logical"] else None
    )
    return {
        "weight_count": len(model.weights),
        "trainable_weight_count": len(model.trainable_weights),
        "logical_parameter_gib": round(bytes_["logical"]/1024**3, 6),
        "sharded_parameter_gib": round(bytes_["sharded"]/1024**3, 6),
        "replicated_parameter_gib": round(bytes_["replicated"]/1024**3, 6),
        "unknown_parameter_gib": round(bytes_["unknown"]/1024**3, 6),
        "sharded_parameter_percent_by_bytes": (
            round(pct, 6) if pct is not None else None
        ),
        "sharded_weight_count": counts["sharded"],
        "replicated_weight_count": counts["replicated"],
        "unknown_sharding_weight_count": counts["unknown"],
        "sharding_samples": samples,
        "byte_weighting_note": (
            "Logical tensor bytes by sharding class; "
            "not exact per-device HBM residency."
        ),
    }

class Gemma4TPUEngine:
    # G3/G5 use native KerasHub generation only for characterization.
    # This is not claimed equivalent to the predecessor split engine.

    def __init__(
        self,
        preset_path: str,
        dtype: str,
        distribution: Any,
        *,
        vision_enabled=True,
        generation_length_buckets=(512,768,1024,1536,2048),
        max_generation_length=2048,
        min_sharded_parameter_percent=80.0,
        phase_callback=None,
        r3_event_callback=None,
    ):
        self.preset_path = preset_path
        self.dtype = dtype
        self.distribution = distribution
        self.vision_enabled = vision_enabled
        self.buckets = tuple(sorted(set(generation_length_buckets)))
        self.max_generation_length = int(max_generation_length)
        self.min_sharded_parameter_percent = float(
            min_sharded_parameter_percent
        )
        self.phase_callback = phase_callback
        self.r3_event_callback = r3_event_callback
        self.model = None
        self.preprocessor = None
        self.metadata = None

    def _phase(self, value):
        if self.phase_callback:
            self.phase_callback(value)

    def load(self):
        import jax, keras, keras_hub

        self._phase("model_construct")
        keras.mixed_precision.set_global_policy(self.dtype)
        path = Path(self.preset_path)
        entry = weights_path(path)
        task_config = path / "task.json"
        started = time.perf_counter()

        # Corrective R1: model.weights.json is a backbone checkpoint.  The
        # KerasHub preset loader knows whether task.json exists and always
        # routes model weights through task.backbone.  Loading the same index
        # directly on the whole Gemma4CausalLM task creates an incompatible
        # object hierarchy and caused the R0 1187-object strict-load failure.
        self._phase("native_preset_strict_load")
        with self.distribution.scope(), sharded_checkpoint_assignment(
            event_callback=self.r3_event_callback,
        ):
            self.model = keras_hub.models.Gemma4CausalLM.from_preset(
                str(path),
                load_weights=True,
                dtype=self.dtype,
            )

        self.model.compile(sampler="greedy", run_eagerly=True)
        self.preprocessor = self.model.preprocessor

        backbone = self.model.backbone
        if self.model.__class__.__name__ != "Gemma4CausalLM":
            raise RuntimeError("Unexpected model class")
        if backbone.__class__.__name__ != "Gemma4Backbone":
            raise RuntimeError("Unexpected backbone class")
        if int(backbone.num_layers) != 60:
            raise RuntimeError(
                f"Expected 60 Gemma 4 decoder layers, "
                f"found {backbone.num_layers}"
            )
        if backbone.vision_encoder is None:
            raise RuntimeError("Expected Gemma 4 31B vision encoder")

        sharding = summarize_sharding(self.model)
        pct = sharding["sharded_parameter_percent_by_bytes"]
        if pct is None or pct < self.min_sharded_parameter_percent:
            raise RuntimeError(
                "Strict load succeeded but sharding coverage failed: "
                f"{pct!r}% < {self.min_sharded_parameter_percent}%"
            )

        self._post_load_host_cleanup()

        self.metadata = {
            "load_seconds": round(time.perf_counter() - started, 6),
            "model": "gemma4_instruct_31b",
            "model_class": self.model.__class__.__name__,
            "backbone_class": backbone.__class__.__name__,
            "num_layers": int(backbone.num_layers),
            "dtype": self.dtype,
            "keras_version": getattr(keras, "__version__", "unknown"),
            "keras_hub_version": getattr(
                keras_hub, "__version__", "unknown"
            ),
            "jax_version": getattr(jax, "__version__", "unknown"),
            "device_count": len(jax.devices("tpu")),
            "devices": [str(d) for d in jax.devices("tpu")],
            "strict_weight_loading": True,
            "skip_mismatch": False,
            "checkpoint_load_strategy": "keras_hub_native_preset_loader",
            "checkpoint_target": "task.backbone",
            "task_config_present": task_config.is_file(),
            "weights_entry": str(entry),
            "layout_profile": LAYOUT_PROFILE,
            "generation_mode": "keras_hub_native_unvalidated",
            "runtime_validation": "NOT_YET_PROVEN",
            **sharding,
            **self.post_load_cleanup,
        }
        self._phase("ready")
        return dict(self.metadata)

    def _post_load_host_cleanup(self):
        self.post_load_cleanup = post_load_host_cleanup()

    def _preprocess_prompt_tokens(self, inputs, sequence_length=None):
        processed = self.preprocessor.generate_preprocess(
            inputs,
            sequence_length=(
                self.max_generation_length
                if sequence_length is None else int(sequence_length)
            ),
        )
        import keras
        mask = keras.ops.convert_to_numpy(processed["padding_mask"])
        return int(mask.sum())

    def _generate(self, inputs, max_tokens, *, authority=False):
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not loaded")
        if authority:
            prompt_tokens = self._preprocess_prompt_tokens(
                inputs, sequence_length=self.max_generation_length
            )
            plan = plan_authority_generation(
                prompt_tokens,
                int(max_tokens),
                self.max_generation_length,
            )
        else:
            prompt_tokens = self._preprocess_prompt_tokens(inputs)
            plan = plan_generation(
                prompt_tokens,
                int(max_tokens),
                self.buckets,
                self.max_generation_length,
            )
        from .observability import CompilationEvidenceCapture

        started = time.perf_counter()
        with CompilationEvidenceCapture() as compilation_capture:
            output = self.model.generate(
                inputs,
                max_length=plan.max_length,
                strip_prompt=True,
            )
        elapsed = time.perf_counter() - started
        return scalar_text(output).strip(), {
            "prompt_tokens": prompt_tokens,
            "requested_new_tokens": int(max_tokens),
            "max_new_tokens": int(max_tokens),
            "generation_max_length": plan.max_length,
            "bucketed": plan.bucketed,
            "generation_seconds": round(elapsed, 6),
            "generation_mode": "keras_hub_native_unvalidated",
            "compile_cache_evidence": compilation_capture.snapshot(),
            "authority_generation_path": (
                "EXACT_LENGTH_AUTHORITY_PATH" if authority else None
            ),
            "authority_max_length": plan.max_length if authority else None,
        }

    def generate_text(self, prompt: str, system: str, max_tokens: int):
        rendered = chat_prompt(prompt, system)
        return self._generate(rendered, max_tokens)

    def generate_text_authority(
        self, prompt: str, system: str, max_tokens: int
    ):
        rendered = chat_prompt(prompt, system)
        return self._generate(rendered, max_tokens, authority=True)

    def generate_image(
        self, image, prompt: str, system: str, max_tokens: int
    ):
        if not self.vision_enabled:
            raise RuntimeError("Vision generation disabled")
        import numpy as np
        rendered = vision_prompt(prompt, system)
        inputs = {"prompts": rendered, "images": np.asarray(image)}
        return self._generate(inputs, max_tokens)
