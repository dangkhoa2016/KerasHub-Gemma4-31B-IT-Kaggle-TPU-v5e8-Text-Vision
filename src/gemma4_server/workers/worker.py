from __future__ import annotations
import logging, os
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger("gemma4_server")
SUPPORTED_V5E8_LABELS = {"v5e-8","v5litepod-8"}

def validate_tpu_fallback(
    require_v5e8: bool, environ: Mapping[str,str]
):
    if not require_v5e8:
        return
    if environ.get("G4_TPU_FALLBACK_APPLIED") != "true":
        return
    accelerator = environ.get("TPU_ACCELERATOR_TYPE", "")
    if accelerator not in SUPPORTED_V5E8_LABELS:
        raise RuntimeError(
            f"Unexpected TPU accelerator type: {accelerator!r}"
        )


def worker_protocol_loop(
    worker_id,
    generation,
    task_queue,
    result_queue,
    shutdown_event,
    loader,
    monitor,
):
    def emit(event_type, **payload):
        result_queue.put({
            "type": event_type,
            "worker_id": worker_id,
            "generation": generation,
            **payload,
        })

    emit("worker_state", state="loading")
    try:
        engine, metadata = loader()
    except Exception as exc:
        emit("worker_load_error", error=repr(exc))
        monitor.stop()
        return

    emit("worker_ready", metadata=metadata)
    while not shutdown_event.is_set():
        task = task_queue.get()
        if task is None:
            break
        emit("job_started", job_id=task["job_id"])
        try:
            if task.get("image") is not None:
                output, metrics = engine.generate_image(
                    task["image"],
                    task["prompt"],
                    task["system"],
                    task["max_tokens"],
                )
            else:
                output, metrics = engine.generate_text(
                    task["prompt"],
                    task["system"],
                    task["max_tokens"],
                )
            emit(
                "job_completed",
                job_id=task["job_id"],
                result=output,
                metrics=metrics,
            )
        except Exception as exc:
            emit("job_failed", job_id=task["job_id"], error=repr(exc))

    monitor.stop()
    emit("worker_stopped", state="stopped")

def model_worker_main(
    worker_id: str,
    generation: int,
    worker_config: dict[str,Any],
    task_queue,
    result_queue,
    shutdown_event,
):
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")

    from ..core.memory import MemoryGuardMonitor
    monitor = MemoryGuardMonitor(
        guard_gib=float(worker_config["memory_guard_gib"]),
        interval_seconds=float(worker_config["memory_poll_seconds"]),
        breach_path=Path("state/memory-guard-breach.json"),
    )
    monitor.start()

    cache_dir = worker_config.get("jax_compilation_cache_dir")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(
            worker_config.get(
                "jax_persistent_cache_min_compile_time_secs", 1
            )
        )
        os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = str(
            worker_config.get(
                "jax_persistent_cache_min_entry_size_bytes", -1
            )
        )

    def loader():
        monitor.set_phase("tpu_import")
        import jax, keras
        from ..tpu.distribution import build_distribution
        from ..tpu.engine import Gemma4TPUEngine

        validate_tpu_fallback(
            bool(worker_config["require_v5e8"]), os.environ
        )
        devices = list(jax.devices("tpu"))
        expected = int(worker_config["expected_tpu_devices"])
        if len(devices) != expected:
            raise RuntimeError(
                f"Expected {expected} TPU devices, found {len(devices)}"
            )

        _mesh, _layout, distribution = build_distribution(
            keras,
            jax,
            shape=tuple(worker_config["mesh_shape"]),
            axis_names=tuple(worker_config["mesh_axis_names"]),
            data_axis=worker_config["data_axis"],
            model_axis=worker_config["model_axis"],
        )

        monitor.set_phase("model_load")
        engine = Gemma4TPUEngine(
            worker_config["model_path"],
            worker_config["model_dtype"],
            distribution,
            vision_enabled=worker_config["vision_enabled"],
            generation_length_buckets=tuple(
                worker_config["generation_length_buckets"]
            ),
            max_generation_length=worker_config[
                "max_generation_length"
            ],
            min_sharded_parameter_percent=worker_config[
                "min_sharded_parameter_percent"
            ],
            phase_callback=monitor.set_phase,
        )
        metadata = engine.load()
        metadata.update({
            "mesh_shape": list(worker_config["mesh_shape"]),
            "mesh_axis_names": list(worker_config["mesh_axis_names"]),
            "backend": "jax",
            "accelerator": "TPU v5e-8",
        })
        return engine, metadata

    worker_protocol_loop(
        worker_id,
        generation,
        task_queue,
        result_queue,
        shutdown_event,
        loader,
        monitor,
    )
