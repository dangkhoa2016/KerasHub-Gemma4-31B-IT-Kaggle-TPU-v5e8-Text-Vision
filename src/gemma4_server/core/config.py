from __future__ import annotations
import math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .model_path import resolve_model_path
from .paths import DATA_DIR, LOG_DIR, STATE_DIR, ARTIFACT_DIR
from .secrets import load_or_create_secret

def _bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1","true","yes","on"}:
        return True
    if value in {"0","false","no","off"}:
        return False
    raise ValueError(f"{name} must be boolean")

def _int(name, default, minimum=None):
    value = int(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value

def _float(name, default, minimum=None):
    value = float(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value

def _ints(name, default):
    return tuple(
        int(v.strip()) for v in os.environ.get(name, default).split(",")
        if v.strip()
    )

def _strs(name, default):
    return tuple(
        v.strip() for v in os.environ.get(name, default).split(",")
        if v.strip()
    )

@dataclass(frozen=True)
class Config:
    model_path: str
    model_preset: str
    host: str
    port: int
    expected_tpu_devices: int
    require_v5e8: bool
    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    data_axis: str
    model_axis: str
    model_dtype: str
    vision_enabled: bool
    generation_mode: str
    generation_length_buckets: tuple[int, ...]
    max_generation_length: int
    default_output_tokens: int
    max_output_tokens: int
    max_input_chars: int
    max_image_bytes: int
    max_image_pixels: int
    max_request_bytes: int
    max_queue_size: int
    max_store_size: int
    result_ttl_seconds: float
    request_timeout: float
    worker_load_timeout: float
    shutdown_timeout: float
    max_worker_restarts: int
    memory_guard_gib: float
    memory_poll_seconds: float
    min_sharded_parameter_percent: float
    jax_compilation_cache_dir: Optional[str]
    jax_persistent_cache_min_compile_time_secs: float
    jax_persistent_cache_min_entry_size_bytes: int
    api_auth_required: bool
    api_key: str
    restart_secret: str

    @classmethod
    def from_env(cls):
        for d in (DATA_DIR, LOG_DIR, STATE_DIR, ARTIFACT_DIR):
            d.mkdir(parents=True, exist_ok=True)

        preset = os.environ.get("MODEL_PRESET", "gemma4_instruct_31b").strip()
        preferred_text = os.environ.get("MODEL_PATH") or os.environ.get(
            "MODEL_BASE",
            "/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b",
        )
        model_path = resolve_model_path(
            Path(preferred_text) if preferred_text else None,
            preset_name=preset,
        )

        mesh_shape = _ints("MESH_SHAPE", "1,8")
        mesh_axes = _strs("MESH_AXIS_NAMES", "batch,model")
        expected = _int("EXPECTED_TPU_DEVICES", 8, 1)
        if len(mesh_shape) != len(mesh_axes) or math.prod(mesh_shape) != expected:
            raise ValueError("Invalid TPU mesh configuration")

        buckets = tuple(sorted(set(_ints(
            "GENERATION_LENGTH_BUCKETS", "16,512,768,1024,1536,2048"
        ))))
        max_generation_length = _int("MAX_GENERATION_LENGTH", 2048, 64)
        if not buckets or buckets[-1] > max_generation_length:
            raise ValueError("Generation buckets must be <= MAX_GENERATION_LENGTH")

        mode = os.environ.get("GENERATION_MODE", "native").strip().lower()
        if mode != "native":
            raise ValueError("Only GENERATION_MODE=native is implemented before G4")

        default_tokens = _int("DEFAULT_OUTPUT_TOKENS", 128, 1)
        max_tokens = _int("MAX_OUTPUT_TOKENS", 512, 1)
        if default_tokens > max_tokens:
            raise ValueError("DEFAULT_OUTPUT_TOKENS must be <= MAX_OUTPUT_TOKENS")

        auth_required = _bool("API_AUTH_REQUIRED", True)
        return cls(
            model_path=str(model_path),
            model_preset=preset,
            host=os.environ.get("HOST", "0.0.0.0"),
            port=_int("PORT", 7860, 1),
            expected_tpu_devices=expected,
            require_v5e8=_bool("REQUIRE_V5E8", True),
            mesh_shape=mesh_shape,
            mesh_axis_names=mesh_axes,
            data_axis=os.environ.get("DATA_PARALLEL_AXIS", "batch"),
            model_axis=os.environ.get("MODEL_PARALLEL_AXIS", "model"),
            model_dtype=os.environ.get("MODEL_DTYPE", "bfloat16"),
            vision_enabled=_bool("VISION_ENABLED", True),
            generation_mode=mode,
            generation_length_buckets=buckets,
            max_generation_length=max_generation_length,
            default_output_tokens=default_tokens,
            max_output_tokens=max_tokens,
            max_input_chars=_int("MAX_INPUT_CHARS", 12000, 1),
            max_image_bytes=_int("MAX_IMAGE_BYTES", 5_242_880, 1024),
            max_image_pixels=_int("MAX_IMAGE_PIXELS", 20_000_000, 64),
            max_request_bytes=_int("MAX_REQUEST_BYTES", 8_388_608, 1024),
            max_queue_size=_int("MAX_QUEUE_SIZE", 8, 1),
            max_store_size=_int("MAX_STORE_SIZE", 500, 1),
            result_ttl_seconds=_float("RESULT_TTL_SECONDS", 3600, 1),
            request_timeout=_float("REQUEST_TIMEOUT", 900, 1),
            worker_load_timeout=_float("WORKER_LOAD_TIMEOUT", 1800, 1),
            shutdown_timeout=_float("SHUTDOWN_TIMEOUT", 300, 1),
            max_worker_restarts=_int("MAX_WORKER_RESTARTS", 1, 0),
            memory_guard_gib=_float("MEMORY_GUARD_GIB", 300, 1),
            memory_poll_seconds=_float("MEMORY_POLL_SECONDS", 1.0, 0.1),
            min_sharded_parameter_percent=_float(
                "MIN_SHARDED_PARAMETER_PERCENT", 80, 0
            ),
            jax_compilation_cache_dir=(
                os.environ.get(
                    "JAX_COMPILATION_CACHE_DIR",
                    "/kaggle/working/.cache/gemma4-31b-jax",
                ).strip() or None
            ),
            jax_persistent_cache_min_compile_time_secs=_float(
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", 1, 0
            ),
            jax_persistent_cache_min_entry_size_bytes=_int(
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", -1, -1
            ),
            api_auth_required=auth_required,
            api_key=load_or_create_secret(
                "API_KEY", DATA_DIR / "api_key.txt", auth_required
            ),
            restart_secret=load_or_create_secret(
                "RESTART_SECRET", DATA_DIR / "restart_secret.txt", True
            ),
        )

    @classmethod
    def for_tests(cls):
        return cls(
            model_path="/tmp/model",
            model_preset="gemma4_instruct_31b",
            host="127.0.0.1",
            port=7860,
            expected_tpu_devices=8,
            require_v5e8=True,
            mesh_shape=(1,8),
            mesh_axis_names=("batch","model"),
            data_axis="batch",
            model_axis="model",
            model_dtype="bfloat16",
            vision_enabled=True,
            generation_mode="native",
            generation_length_buckets=(16,512,768,1024),
            max_generation_length=1024,
            default_output_tokens=64,
            max_output_tokens=256,
            max_input_chars=2000,
            max_image_bytes=524288,
            max_image_pixels=20_000_000,
            max_request_bytes=2_000_000,
            max_queue_size=4,
            max_store_size=20,
            result_ttl_seconds=60,
            request_timeout=0.05,
            worker_load_timeout=5,
            shutdown_timeout=1,
            max_worker_restarts=1,
            memory_guard_gib=300,
            memory_poll_seconds=1,
            min_sharded_parameter_percent=80,
            jax_compilation_cache_dir=None,
            jax_persistent_cache_min_compile_time_secs=1,
            jax_persistent_cache_min_entry_size_bytes=-1,
            api_auth_required=True,
            api_key="test-api-key",
            restart_secret="test-restart-secret",
        )

    def worker_payload(self):
        keys = (
            "model_path","model_preset","expected_tpu_devices","require_v5e8",
            "mesh_shape","mesh_axis_names","data_axis","model_axis","model_dtype",
            "vision_enabled","generation_mode","generation_length_buckets",
            "max_generation_length","memory_guard_gib","memory_poll_seconds",
            "min_sharded_parameter_percent","jax_compilation_cache_dir",
            "jax_persistent_cache_min_compile_time_secs",
            "jax_persistent_cache_min_entry_size_bytes",
        )
        return {k: getattr(self, k) for k in keys}
