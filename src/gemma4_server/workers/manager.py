from __future__ import annotations
import logging, multiprocessing as mp, queue, threading, time, uuid

from ..core.errors import QueueFullError, WorkerNotReadyError
from ..jobs.models import Job
from ..jobs.store import JobStore

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
        self._collector_stop = threading.Event()
        self._shutting_down = threading.Event()
        self._accepting = True
        self._restart_pending = False
        self._collector_thread = None

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

    def submit(self, payload, request_id=None):
        if (
            not self._accepting
            or self._shutting_down.is_set()
        ):
            raise WorkerNotReadyError(
                health=self.health()
            )

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
            shutting_down = self._shutting_down.is_set()

        return {
            "state": (
                "unavailable"
                if not accepting or shutting_down
                else "ready"
            ),
            "ready": (
                accepting and not shutting_down
            ),
            "ready_workers": 0,
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
            "runtime": {
                "model": "gemma4_instruct_31b",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": (
                    self.config.expected_tpu_devices
                ),
                "mesh": list(self.config.mesh_shape),
                "runtime_validation": "NOT_YET_PROVEN",
            },
        }

    def wait_idle(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.store.pending_count() == 0:
                return True
            time.sleep(0.2)
        return self.store.pending_count() == 0
