from __future__ import annotations

import sys
import queue
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.api.app import Runtime, create_app
from gemma4_server.core.config import Config
from gemma4_server.jobs.models import Job
from gemma4_server.jobs.store import JobStore
from gemma4_server.workers.manager import GenerationManager


class AsyncJobStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = JobStore(4, 60)

    def test_failed_job_cannot_be_completed_by_late_worker_event(self):
        job = Job("job-123", "prompt", "", 16)
        self.store.put(job)
        self.store.mark_failed(job.id, "TPU worker restarted")

        self.store.mark_completed(job.id, "late output", 1.0)

        stored = self.store.get(job.id)
        self.assertEqual(stored.status, "failed")
        self.assertEqual(stored.public_error, "TPU worker restarted")
        self.assertIsNone(stored.result)

    def test_completed_result_survives_restart(self):
        job = Job("job-123", "prompt", "", 16)
        self.store.put(job)
        self.store.mark_completed(job.id, "completed output", 1.0)

        self.store.fail_pending("TPU worker restarted")

        stored = self.store.get(job.id)
        self.assertEqual(stored.status, "completed")
        self.assertEqual(stored.public_dict()["output"], "completed output")


class FakeQueue:
    def __init__(self):
        self.items = []
        self.closed = False

    def put_nowait(self, item):
        self.items.append(item)

    def get(self, timeout=None):
        raise queue.Empty

    def close(self):
        self.closed = True


class FakeContext:
    def Queue(self, maxsize=None):
        return FakeQueue()

    def Event(self):
        return threading.Event()


class FakeProcess:
    def __init__(self):
        self.pid = 123
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.alive = False

    def terminate(self):
        self.alive = False


class AsyncManagerTests(unittest.TestCase):
    API_HEADERS = {"Authorization": "Bearer test-api-key"}

    def setUp(self):
        self.manager = GenerationManager(Config.for_tests())
        self.manager.ctx = FakeContext()
        self.manager.task_queue = FakeQueue()
        self.manager.result_queue = FakeQueue()
        self.manager.shutdown_event = threading.Event()
        self.manager._clear_pid = lambda expected=None: None
        self.manager._generation = 2
        self.manager._worker = FakeProcess()
        self.manager._worker_status[self.manager.WORKER_ID] = {
            "worker_id": self.manager.WORKER_ID,
            "generation": 2,
            "state": "ready",
            "metadata": {"device_count": 8},
        }
        self.starts = []

        def start_worker():
            self.manager._generation += 1
            self.manager._worker = FakeProcess()
            self.manager._worker_status[self.manager.WORKER_ID] = {
                "worker_id": self.manager.WORKER_ID,
                "generation": self.manager._generation,
                "state": "starting",
            }
            self.starts.append(self.manager._generation)

        self.manager._start_worker = start_worker
        self.addCleanup(self.manager._collector_stop.set)

    def put_job(self, job_id="job-123"):
        job = Job(job_id, "prompt", "", 16)
        self.manager.store.put(job)
        return job

    def test_stale_old_generation_cannot_complete_failed_job(self):
        job = self.put_job()
        self.manager.store.mark_failed(job.id, "TPU worker restarted")

        self.manager._handle({
            "type": "job_completed",
            "worker_id": self.manager.WORKER_ID,
            "generation": 1,
            "job_id": job.id,
            "result": "stale output",
            "inference_seconds": 1.0,
        })
        self.manager._handle({
            "type": "job_completed",
            "worker_id": self.manager.WORKER_ID,
            "generation": 2,
            "job_id": job.id,
            "result": "current output",
            "inference_seconds": 1.0,
        })

        stored = self.manager.store.get(job.id)
        self.assertEqual(stored.status, "failed")
        self.assertEqual(stored.public_error, "TPU worker restarted")
        self.assertIsNone(stored.result)

    def test_restart_without_wait_fails_pending_jobs(self):
        queued = self.put_job("queued")
        processing = self.put_job("processing")
        self.manager.store.mark_processing(processing.id, self.manager.WORKER_ID)

        restarted_idle = self.manager.restart_worker(False, 1)

        self.assertFalse(restarted_idle)
        self.assertEqual(self.manager.store.get(queued.id).status, "failed")
        self.assertEqual(
            self.manager.store.get(processing.id).status, "failed"
        )
        self.assertEqual(self.starts, [3])

    def test_restart_waits_for_processing_job_when_requested(self):
        job = self.put_job()
        self.manager.store.mark_processing(job.id, self.manager.WORKER_ID)
        waiting = threading.Event()
        release = threading.Event()
        results = []

        def wait_idle(timeout):
            waiting.set()
            release.wait(1)
            return self.manager.store.pending_count() == 0

        self.manager.wait_idle = wait_idle
        restart = threading.Thread(
            target=lambda: results.append(self.manager.restart_worker(True, 1))
        )
        restart.start()
        self.assertTrue(waiting.wait(1))
        self.assertEqual(self.manager.store.get(job.id).status, "processing")

        self.manager.store.mark_completed(job.id, "output", 1.0)
        release.set()
        restart.join(1)

        self.assertEqual(results, [True])
        self.assertEqual(self.manager.store.get(job.id).status, "completed")
        self.assertEqual(self.starts, [3])

    def test_old_completion_after_replacement_generation_is_current_is_ignored(self):
        job = self.put_job()
        self.manager.store.mark_processing(job.id, self.manager.WORKER_ID)
        replacement_current = threading.Event()
        restart_finished = threading.Event()
        results = []

        def restart_worker():
            results.append(self.manager.restart_worker(False, 1))
            replacement_current.set()
            restart_finished.set()

        restart = threading.Thread(target=restart_worker)
        restart.start()
        self.assertTrue(replacement_current.wait(timeout=1))
        self.assertEqual(self.manager._generation, 3)
        self.manager._handle({
            "type": "job_completed", "worker_id": self.manager.WORKER_ID,
            "generation": 2, "job_id": job.id, "result": "old output",
            "inference_seconds": 1.0,
        })
        self.assertTrue(restart_finished.is_set())
        restart.join(timeout=1)

        stored = self.manager.store.get(job.id)
        self.assertEqual(results, [False])
        self.assertEqual(
            self.manager._worker_status[self.manager.WORKER_ID]["state"],
            "starting",
        )
        self.assertEqual(stored.status, "failed")
        self.assertIsNone(stored.result)

    def test_async_polling_reports_queued_processing_and_terminal_states(self):
        job = self.put_job()
        app = create_app(Runtime(Config.for_tests(), self.manager))
        client = app.test_client()

        queued = client.get(f"/result/{job.id}", headers=self.API_HEADERS)
        self.manager.store.mark_processing(job.id, self.manager.WORKER_ID)
        processing = client.get(f"/result/{job.id}", headers=self.API_HEADERS)
        self.manager.store.mark_completed(job.id, "output", 1.0)
        completed = client.get(f"/result/{job.id}", headers=self.API_HEADERS)

        self.assertEqual((queued.status_code, queued.get_json()["status"]), (202, "queued"))
        self.assertEqual(
            (processing.status_code, processing.get_json()["status"]),
            (202, "processing"),
        )
        self.assertEqual(
            (completed.status_code, completed.get_json()["status"]),
            (200, "completed"),
        )
        self.assertEqual(completed.get_json()["output"], "output")

    def test_generation_failure_is_http_500_with_json_error(self):
        job = self.put_job()
        self.manager._handle({
            "type": "job_failed",
            "worker_id": self.manager.WORKER_ID,
            "generation": 2,
            "job_id": job.id,
            "error": "worker generation exception",
        })
        app = create_app(Runtime(Config.for_tests(), self.manager))
        client = app.test_client()

        response = client.get(f"/result/{job.id}", headers=self.API_HEADERS)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.get_json()["status"], "failed")
        self.assertEqual(response.get_json()["error"], "Generation failed")


if __name__ == "__main__":
    unittest.main()
