from __future__ import annotations
import threading, time
from collections import OrderedDict
from ..core.errors import StoreFullError
from .models import Job

class JobStore:
    def __init__(self, max_size: int, ttl_seconds: float):
        self._store = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _cleanup(self):
        now = time.time()
        for key, job in list(self._store.items()):
            if job.status in {"completed","failed"} and job.completed_at:
                if now - job.completed_at >= self._ttl:
                    self._store.pop(key, None)

    def put(self, job: Job):
        with self._lock:
            self._cleanup()
            while len(self._store) >= self._max_size:
                removable = next(
                    (k for k,v in self._store.items()
                     if v.status in {"completed","failed"}), None
                )
                if removable is None:
                    raise StoreFullError("Job result store is full")
                self._store.pop(removable)
            self._store[job.id] = job

    def delete(self, job_id):
        with self._lock:
            self._store.pop(job_id, None)

    def get(self, job_id):
        with self._lock:
            self._cleanup()
            return self._store.get(job_id)

    def mark_processing(self, job_id, worker_id):
        with self._lock:
            job = self._store.get(job_id)
            if job and job.status == "queued":
                job.status = "processing"
                job.started_at = time.time()
                job.worker_id = worker_id

    def mark_completed(self, job_id, result, inference_seconds, metrics=None, runtime=None):
        with self._lock:
            job = self._store.get(job_id)
            if not job or job.status not in {"queued", "processing"}:
                return
            job.status = "completed"
            job.result = result
            job.inference_seconds = inference_seconds
            job.metrics = metrics or {}
            job.runtime = runtime or {}
            job.image = None
            job.completed_at = time.time()
            job.done.set()

    def mark_failed(self, job_id, public_error, internal_error=None):
        with self._lock:
            job = self._store.get(job_id)
            if not job or job.status not in {"queued", "processing"}:
                return
            job.status = "failed"
            job.public_error = public_error
            job.internal_error = internal_error
            job.image = None
            job.completed_at = time.time()
            job.done.set()

    def fail_pending(self, public_error, internal_error=None):
        with self._lock:
            ids = [
                j.id for j in self._store.values()
                if j.status in {"queued","processing"}
            ]
            for job_id in ids:
                self.mark_failed(job_id, public_error, internal_error)
            return len(ids)

    def pending_count(self):
        with self._lock:
            return sum(
                j.status in {"queued","processing"} for j in self._store.values()
            )

    def stats(self):
        with self._lock:
            self._cleanup()
            out = {"queued":0,"processing":0,"completed":0,"failed":0}
            for job in self._store.values():
                out[job.status] += 1
            out["total"] = len(self._store)
            return out
