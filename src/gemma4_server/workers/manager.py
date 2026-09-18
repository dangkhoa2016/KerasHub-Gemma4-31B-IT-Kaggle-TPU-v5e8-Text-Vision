from __future__ import annotations
import logging, multiprocessing as mp, os, queue, threading, time, uuid
from pathlib import Path

from ..core.errors import QueueFullError, WorkerNotReadyError
from ..core.paths import STATE_DIR
from ..jobs.models import Job
from ..jobs.store import JobStore
from .worker import model_worker_main

logger = logging.getLogger("gemma4_server")

class GenerationManager:
    WORKER_ID = "tpu-0"

    def __init__(self, config):
        self.config = config
        self.store = JobStore(
            config.max_store_size, config.result_ttl_seconds
        )
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(
            maxsize=config.max_queue_size
        )
        self.result_queue = self.ctx.Queue()
        self.shutdown_event = self.ctx.Event()
        self._worker = None
        self._generation = 0
        self._worker_status = {}
        self._automatic_restarts_used = 0
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._collector_stop = threading.Event()
        self._shutting_down = threading.Event()
        self._accepting = True
        self._restart_pending = False
        self._collector_thread = None

    @property
    def worker_pid_file(self):
        return STATE_DIR / "worker.pid"

    def _write_pid(self, process):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.worker_pid_file.write_text(
            f"{process.pid}\n", encoding="utf-8"
        )

    def _clear_pid(self, expected=None):
        try:
            if expected is not None:
                current = int(
                    self.worker_pid_file.read_text().strip()
                )
                if current != expected:
                    return
            self.worker_pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    def start_async(self):
        with self._lifecycle_lock:
            with self._lock:
                if self._shutting_down.is_set():
                    return False
                status = self._worker_status.get(self.WORKER_ID, {})
                if (
                    self._worker is not None
                    or status.get("state") in {
                        "starting", "loading", "ready", "busy"
                    }
                ):
                    return False
                if self._collector_thread is None:
                    self._collector_thread = threading.Thread(
                        target=self._collect,
                        name="tpu-results",
                        daemon=True,
                    )
                    self._collector_thread.start()
            self._start_worker()
            return True

    def _start_worker(self):
        self._generation += 1
        generation = self._generation
        with self._lock:
            self._worker_status[self.WORKER_ID] = {
                "worker_id": self.WORKER_ID,
                "generation": generation,
                "state": "starting",
                "started_at": time.time(),
            }

        process = self.ctx.Process(
            target=model_worker_main,
            args=(
                self.WORKER_ID,
                generation,
                self.config.worker_payload(),
                self.task_queue,
                self.result_queue,
                self.shutdown_event,
            ),
            name=f"gemma4-{self.WORKER_ID}",
        )
        process.start()
        self._worker = process
        self._write_pid(process)
        with self._lock:
            self._worker_status[self.WORKER_ID][
                "pid"
            ] = process.pid

        threading.Thread(
            target=self._monitor,
            args=(process, generation),
            name="tpu-monitor",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._load_watchdog,
            args=(process, generation),
            name="tpu-load-watchdog",
            daemon=True,
        ).start()

    def _load_watchdog(self, process, generation):
        deadline = (
            time.monotonic() + self.config.worker_load_timeout
        )
        while (
            time.monotonic() < deadline
            and not self._shutting_down.is_set()
        ):
            with self._lock:
                status = self._worker_status.get(
                    self.WORKER_ID, {}
                )
                if status.get("generation") != generation:
                    return
                if status.get("state") not in {
                    "starting","loading"
                }:
                    return
            time.sleep(0.25)

        if (
            process.is_alive()
            and not self._shutting_down.is_set()
        ):
            with self._lock:
                status = self._worker_status.get(
                    self.WORKER_ID, {}
                )
                if status.get("generation") == generation:
                    status.update(
                        state="failed",
                        error=(
                            "Worker load timeout after "
                            f"{self.config.worker_load_timeout}s"
                        ),
                    )
            process.terminate()

    def _collect(self):
        while not self._collector_stop.is_set():
            try:
                message = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                continue
            self._handle(message)

    def _handle(self, message):
        worker_id = message.get(
            "worker_id", self.WORKER_ID
        )
        with self._lock:
            status = self._worker_status.setdefault(
                worker_id, {}
            )
            if message.get("generation") != status.get(
                "generation"
            ):
                return
            kind = message.get("type")
            if (
                kind == "worker_ready"
                and status.get("state") not in {"starting", "loading"}
            ):
                return
            status["pid"] = message.get("pid")
            if kind == "worker_state":
                status["state"] = message["state"]
            elif kind == "worker_ready":
                status.update(
                    state="ready",
                    metadata=message.get("metadata", {}),
                )
                self._restart_pending = False
                self._accepting = True
            elif kind == "worker_load_error":
                status.update(
                    state="failed",
                    error=message.get("error"),
                )
                self._restart_pending = False
                self._accepting = False
            elif kind == "job_started":
                status["state"] = "busy"
                status["active_job_id"] = message["job_id"]
                self.store.mark_processing(
                    message["job_id"], worker_id
                )
            elif kind == "job_completed":
                status["state"] = "ready"
                status.pop("active_job_id", None)
                metrics = message.get("metrics") or {}
                self.store.mark_completed(
                    message["job_id"],
                    message["result"],
                    message.get(
                        "inference_seconds",
                        metrics.get("generation_seconds"),
                    ),
                    metrics,
                    message.get("runtime"),
                )
            elif kind == "job_failed":
                status["state"] = "ready"
                status.pop("active_job_id", None)
                self.store.mark_failed(
                    message["job_id"],
                    "Generation failed",
                    message.get("error"),
                )
            elif kind == "worker_stopped":
                status["state"] = "stopped"

    def _invalidate_current_worker_locked(self):
        status = self._worker_status.get(self.WORKER_ID)
        if status:
            status["state"] = "stopping"
            status["generation"] = None

    def _monitor(self, process, generation):
        process.join()
        self._clear_pid(process.pid)
        if self._shutting_down.is_set():
            return

        with self._lock:
            status = self._worker_status.get(
                self.WORKER_ID, {}
            )
            if status.get("generation") != generation:
                return
            if status.get("state") not in {
                "stopped","failed"
            }:
                status["state"] = "failed"
                status["error"] = (
                    f"Worker exited with code {process.exitcode}"
                )

        with self._lifecycle_lock:
            if self._shutting_down.is_set():
                return
            with self._lock:
                status = self._worker_status.get(
                    self.WORKER_ID, {}
                )
                if status.get("generation") != generation:
                    return
                if (
                    self._automatic_restarts_used
                    >= self.config.max_worker_restarts
                ):
                    self._accepting = False
                    return
                self._automatic_restarts_used += 1
                self._accepting = False
                self._restart_pending = True
                self._invalidate_current_worker_locked()
                self._worker = None
            self.store.fail_pending(
                "TPU worker stopped unexpectedly"
            )
            self._replace_worker_resources()
            self._start_worker()

    def submit(self, payload, request_id=None):
        health = self.health()
        if (
            not health["ready"]
            or not health["accepting_jobs"]
        ):
            raise WorkerNotReadyError(health=health)

        job = Job(
            id=f"job-{uuid.uuid4().hex[:24]}",
            prompt=payload["prompt"],
            system=payload.get("system",""),
            max_tokens=int(payload["max_tokens"]),
            request_id=request_id,
            image=payload.get("image"),
        )
        self.store.put(job)
        try:
            self.task_queue.put_nowait({
                "job_id": job.id,
                "prompt": job.prompt,
                "system": job.system,
                "max_tokens": job.max_tokens,
                "image": job.image,
            })
        except queue.Full:
            self.store.delete(job.id)
            raise QueueFullError("Inference queue is full")
        return job

    def health(self):
        with self._lock:
            workers = [
                dict(v)
                for v in self._worker_status.values()
            ]
            accepting = self._accepting
            restart_pending = self._restart_pending
            shutting_down = self._shutting_down.is_set()

        ready_workers = sum(
            w.get("state") in {"ready","busy"}
            and (w.get("metadata") or {}).get(
                "device_count"
            ) == self.config.expected_tpu_devices
            for w in workers
        )

        ready = (
            ready_workers == 1
            and accepting
            and not restart_pending
            and not shutting_down
        )
        if restart_pending:
            state = "restarting"
        elif shutting_down:
            state = "unavailable"
        elif not accepting:
            state = (
                "unavailable"
                if any(
                    w.get("state") in {"failed", "stopped"}
                    for w in workers
                )
                else "loading"
            )
        elif ready:
            state = "ready"
        elif (
            not workers
            or any(
                w.get("state") in {"starting","loading"}
                for w in workers
            )
        ):
            state = "loading"
        else:
            state = "unavailable"

        runtime = {
            "model": "gemma4_instruct_31b",
            "backend": "jax",
            "accelerator": "TPU v5e-8",
            "source_sha": os.environ.get("FINAL_TPU_EXECUTION_SHA"),
            "expected_tpu_devices": (
                self.config.expected_tpu_devices
            ),
            "mesh": list(self.config.mesh_shape),
            "runtime_validation": "NOT_YET_PROVEN",
        }
        ready_metadata = next(
            (
                w.get("metadata") or {}
                for w in workers
                if w.get("state") in {"ready", "busy"}
                and (w.get("metadata") or {}).get("device_count")
                == self.config.expected_tpu_devices
            ),
            {},
        )
        for key in (
            "jax_default_backend",
            "dtype",
            "keras_version",
            "keras_hub_version",
            "jax_version",
            "device_count",
            "devices",
            "model_class",
            "backbone_class",
            "num_layers",
            "mesh_shape",
            "mesh_axis_names",
            "strict_weight_loading",
            "skip_mismatch",
            "checkpoint_load_strategy",
            "layout_profile",
            "candidate_a_verified",
            "sharded_parameter_percent_by_bytes",
        ):
            if key in ready_metadata:
                runtime[key] = ready_metadata[key]

        return {
            "state": state,
            "ready": ready,
            "ready_workers": ready_workers,
            "expected_workers": 1,
            "accepting_jobs": (
                accepting and not shutting_down
            ),
            "worker_generation": self._generation,
            "automatic_restarts_used": (
                self._automatic_restarts_used
            ),
            "jobs": self.store.stats(),
            "workers": workers,
            "runtime": runtime,
        }

    def wait_idle(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.store.pending_count() == 0:
                return True
            time.sleep(0.2)
        return self.store.pending_count() == 0

    def _stop_worker(self, timeout):
        process = self._worker
        if not process:
            return
        try:
            self.task_queue.put_nowait(None)
        except Exception:
            pass
        process.join(timeout=min(max(timeout, 0.1), 30))
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            killer = getattr(
                process, "kill", process.terminate
            )
            killer()
            process.join(timeout=5)
        if process.is_alive():
            raise RuntimeError(
                "TPU worker did not stop"
            )
        self._clear_pid(process.pid)
        self._worker = None

    def restart_worker(
        self, wait_for_jobs=True, timeout=300
    ):
        if not self._lifecycle_lock.acquire(blocking=False):
            return False
        try:
            if self._shutting_down.is_set():
                return False
            with self._lock:
                if self._restart_pending:
                    return False
                self._accepting = False
                self._restart_pending = True
                self._invalidate_current_worker_locked()
            idle = (
                self.wait_idle(timeout)
                if wait_for_jobs
                else self.store.pending_count() == 0
            )
            if not idle:
                self.store.fail_pending(
                    "TPU worker restarted"
                )

            self._stop_worker(timeout)
            self._replace_worker_resources()
            self._start_worker()
            return idle
        finally:
            self._lifecycle_lock.release()

    def _replace_worker_resources(self):
        self.shutdown_event = self.ctx.Event()
        try:
            self.task_queue.close()
            self.result_queue.close()
        except Exception:
            pass
        self.task_queue = self.ctx.Queue(
            maxsize=self.config.max_queue_size
        )
        self.result_queue = self.ctx.Queue()

    def shutdown(
        self, wait_for_jobs=False, timeout=300
    ):
        with self._lifecycle_lock:
            if self._shutting_down.is_set():
                return True
            self._shutting_down.set()
            with self._lock:
                self._accepting = False
                self._restart_pending = False
                status = self._worker_status.get(self.WORKER_ID)
                if status:
                    status["state"] = "stopping"
            idle = (
                self.wait_idle(timeout)
                if wait_for_jobs else True
            )
            self.shutdown_event.set()
            self._stop_worker(timeout)
            self._collector_stop.set()
            if self._collector_thread:
                self._collector_thread.join(timeout=2)
            return idle
