from __future__ import annotations

import queue
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.core.config import Config
from gemma4_server.jobs.models import Job
from gemma4_server.workers.manager import GenerationManager


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
    _next_pid = 1000

    def __init__(self, events, label):
        self.events = events
        self.label = label
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.exitcode = 0
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.events.append(("join", self.label))
        self.alive = False

    def terminate(self):
        self.events.append(("terminate", self.label))
        self.alive = False


class TrackingLifecycleLock:
    def __init__(self):
        self._lock = threading.RLock()
        self.owner = None
        self.contender_entered = threading.Event()

    def acquire(self, blocking=True):
        if (
            self.owner is not None
            and self.owner != threading.get_ident()
        ):
            self.contender_entered.set()
        acquired = self._lock.acquire(blocking)
        if acquired:
            self.owner = threading.get_ident()
        return acquired

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()


class ManagerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.manager = GenerationManager(Config.for_tests())
        self.manager.ctx = FakeContext()
        self.manager.task_queue = FakeQueue()
        self.manager.result_queue = FakeQueue()
        self.manager.shutdown_event = threading.Event()
        self.events = []
        self.starts = []
        self.addCleanup(self.manager._collector_stop.set)

    def install_start_fake(self, block_first_start=False):
        entered = threading.Event()
        release = threading.Event()

        def start_worker():
            generation = self.manager._generation + 1
            self.manager._generation = generation
            process = FakeProcess(self.events, f"worker-{generation}")
            self.manager._worker = process
            self.manager._worker_status[self.manager.WORKER_ID] = {
                "worker_id": self.manager.WORKER_ID,
                "generation": generation,
                "state": "starting",
            }
            self.starts.append(process)
            self.events.append(("start", process.label))
            if block_first_start and len(self.starts) == 1:
                entered.set()
                release.wait(2)

        self.manager._start_worker = start_worker
        return entered, release

    def install_existing_worker(self):
        process = FakeProcess(self.events, "old")
        self.manager._worker = process
        self.manager._generation = 1
        self.manager._worker_status[self.manager.WORKER_ID] = {
            "worker_id": self.manager.WORKER_ID,
            "generation": 1,
            "state": "ready",
            "metadata": {"device_count": 8},
        }
        return process

    def run_in_thread(self, barrier, target, results):
        def run():
            barrier.wait()
            results.append(target())

        thread = threading.Thread(target=run)
        thread.start()
        return thread

    def test_start_async_is_idempotent_during_loading(self):
        entered, release = self.install_start_fake(block_first_start=True)
        results = []
        first = threading.Thread(
            target=lambda: results.append(self.manager.start_async())
        )
        first.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(self.manager.start_async())
        release.set()
        first.join(1)

        self.assertEqual(results, [True])
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(self.manager._generation, 1)

    def test_start_and_restart_race_has_one_worker_owner(self):
        entered, release = self.install_start_fake(block_first_start=True)
        self.manager._lifecycle_lock = TrackingLifecycleLock()
        results = []
        start = threading.Thread(
            target=lambda: results.append(self.manager.start_async())
        )
        start.start()
        self.assertTrue(entered.wait(1))
        restart = threading.Thread(
            target=lambda: results.append(
                self.manager.restart_worker(
                wait_for_jobs=False, timeout=1
                )
            )
        )
        restart.start()
        self.assertTrue(
            self.manager._lifecycle_lock.contender_entered.wait(1)
        )
        release.set()
        start.join(1)
        restart.join(1)

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 1)
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(self.manager._generation, 1)

    def test_two_simultaneous_restarts_have_one_transition(self):
        self.install_existing_worker()
        entered, release = self.install_start_fake(block_first_start=True)
        barrier = threading.Barrier(2)
        results = []
        first = self.run_in_thread(
            barrier,
            lambda: self.manager.restart_worker(False, 1),
            results,
        )
        second = self.run_in_thread(
            barrier,
            lambda: self.manager.restart_worker(False, 1),
            results,
        )
        self.assertTrue(entered.wait(1))
        release.set()
        first.join(1)
        second.join(1)

        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(self.manager._generation, 2)

    def test_restart_rejected_while_replacement_is_still_loading(self):
        self.install_existing_worker()
        self.install_start_fake()
        replacement_started = threading.Event()
        first_returned = threading.Event()
        original_start = self.manager._start_worker

        def start_worker_and_release_loading_gate():
            original_start()
            replacement_started.set()

        self.manager._start_worker = start_worker_and_release_loading_gate
        results = []

        def first_restart():
            results.append((
                "first",
                self.manager.restart_worker(False, 1),
            ))
            first_returned.set()

        first = threading.Thread(target=first_restart)
        first.start()
        self.assertTrue(replacement_started.wait(timeout=1))
        self.assertTrue(first_returned.wait(timeout=1))

        second = threading.Thread(
            target=lambda: results.append((
                "second",
                self.manager.restart_worker(False, 1),
            ))
        )
        second.start()
        second.join(timeout=1)
        first.join(timeout=1)

        self.assertEqual(dict(results), {"first": True, "second": False})
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(self.manager._generation, 2)
        self.assertTrue(self.manager._restart_pending)
        self.assertFalse(self.manager.health()["ready"])

    def test_health_is_not_ready_during_restart(self):
        self.install_existing_worker()
        self.manager._restart_pending = True

        health = self.manager.health()

        self.assertEqual(health["state"], "restarting")
        self.assertFalse(health["ready"])

    def test_health_is_not_ready_during_loading(self):
        self.install_existing_worker()
        self.manager._accepting = False

        health = self.manager.health()

        self.assertEqual(health["state"], "loading")
        self.assertFalse(health["ready"])

    def test_health_becomes_ready_only_after_worker_ready(self):
        self.manager._generation = 3
        self.manager._worker_status[self.manager.WORKER_ID] = {
            "worker_id": self.manager.WORKER_ID,
            "generation": 3,
            "state": "loading",
        }
        self.manager._accepting = False
        self.manager._restart_pending = True

        self.manager._handle({
            "type": "worker_ready", "worker_id": self.manager.WORKER_ID,
            "generation": 3, "pid": 123, "metadata": {"device_count": 8},
        })

        health = self.manager.health()
        self.assertEqual(health["state"], "ready")
        self.assertTrue(health["ready"])

    def test_job_completed_event_uses_metrics_when_inference_seconds_is_nested(self):
        job = Job(
            id="job-1",
            prompt="Hello",
            system="",
            max_tokens=1,
            request_id="request-1",
        )
        self.manager.store.put(job)
        self.manager._generation = 1
        self.manager._worker_status[self.manager.WORKER_ID] = {
            "worker_id": self.manager.WORKER_ID,
            "generation": 1,
            "state": "ready",
        }

        self.manager._handle({
            "type": "job_started",
            "worker_id": self.manager.WORKER_ID,
            "generation": 1,
            "pid": 123,
            "job_id": job.id,
        })
        self.manager._handle({
            "type": "job_completed",
            "worker_id": self.manager.WORKER_ID,
            "generation": 1,
            "pid": 123,
            "job_id": job.id,
            "result": "Hi",
            "metrics": {
                "generation_seconds": 1.25,
                "prompt_tokens": 10,
                "generation_max_length": 16,
            },
        })

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.result, "Hi")
        self.assertEqual(job.inference_seconds, 1.25)
        self.assertEqual(job.metrics["generation_max_length"], 16)

    def test_stopping_generation_ready_event_does_not_reopen_readiness(self):
        self.install_existing_worker()
        status = self.manager._worker_status[self.manager.WORKER_ID]
        status["state"] = "stopping"
        self.manager._accepting = False
        self.manager._restart_pending = True

        self.manager._handle({
            "type": "worker_ready", "worker_id": self.manager.WORKER_ID,
            "generation": 1, "pid": 123, "metadata": {"device_count": 8},
        })

        health = self.manager.health()
        self.assertEqual(status["state"], "stopping")
        self.assertEqual(health["state"], "restarting")
        self.assertFalse(health["ready"])
        self.assertFalse(health["accepting_jobs"])

    def test_restart_invalidates_stopping_generation_before_queued_events(self):
        self.install_existing_worker()
        self.install_start_fake()

        def stop_worker(timeout):
            self.manager._handle({
                "type": "worker_state",
                "worker_id": self.manager.WORKER_ID,
                "generation": 1,
                "pid": 123,
                "state": "loading",
            })
            self.manager._handle({
                "type": "worker_ready",
                "worker_id": self.manager.WORKER_ID,
                "generation": 1,
                "pid": 123,
                "metadata": {"device_count": 8},
            })
            self.manager._worker = None

        self.manager._stop_worker = stop_worker

        self.assertTrue(self.manager.restart_worker(False, 1))

        health = self.manager.health()
        self.assertEqual(self.manager._generation, 2)
        self.assertEqual(health["state"], "restarting")
        self.assertFalse(health["ready"])
        self.assertFalse(health["accepting_jobs"])

    def test_replacement_load_failure_leaves_manager_unavailable(self):
        self.install_existing_worker()
        replacement_started = threading.Event()
        release_replacement_start = threading.Event()

        def start_worker():
            self.manager._generation += 1
            generation = self.manager._generation
            self.manager._worker = FakeProcess(self.events, f"worker-{generation}")
            self.manager._worker_status[self.manager.WORKER_ID] = {
                "worker_id": self.manager.WORKER_ID,
                "generation": generation,
                "state": "starting",
            }
            self.starts.append(self.manager._worker)
            replacement_started.set()
            self.assertTrue(release_replacement_start.wait(timeout=1))

        self.manager._start_worker = start_worker
        result = []
        restart = threading.Thread(
            target=lambda: result.append(self.manager.restart_worker(False, 1))
        )
        restart.start()
        self.assertTrue(replacement_started.wait(timeout=1))
        self.manager._handle({
            "type": "worker_load_error", "worker_id": self.manager.WORKER_ID,
            "generation": 2, "pid": 123, "error": "replacement load failed",
        })
        release_replacement_start.set()
        restart.join(timeout=1)

        health = self.manager.health()
        self.assertEqual(result, [True])
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(health["state"], "unavailable")
        self.assertFalse(health["ready"])
        self.assertFalse(health["accepting_jobs"])

    def test_restart_during_busy_fails_job_before_replacement_starts(self):
        self.install_existing_worker()
        self.manager.store.put(Job("busy-job", "prompt", "", 16))
        self.manager.store.mark_processing("busy-job", self.manager.WORKER_ID)
        stop_entered = threading.Event()
        release_stop = threading.Event()
        replacement_started = threading.Event()

        def stop_worker(timeout):
            stop_entered.set()
            self.assertTrue(release_stop.wait(timeout=1))
            self.manager._worker = None

        def start_worker():
            self.manager._generation += 1
            generation = self.manager._generation
            self.manager._worker = FakeProcess(self.events, f"worker-{generation}")
            self.manager._worker_status[self.manager.WORKER_ID] = {
                "worker_id": self.manager.WORKER_ID,
                "generation": generation,
                "state": "starting",
            }
            self.starts.append(self.manager._worker)
            replacement_started.set()

        self.manager._stop_worker = stop_worker
        self.manager._start_worker = start_worker
        result = []
        restart = threading.Thread(
            target=lambda: result.append(self.manager.restart_worker(False, 1))
        )
        restart.start()
        self.assertTrue(stop_entered.wait(timeout=1))
        self.assertEqual(self.manager.store.get("busy-job").status, "failed")
        self.assertEqual(self.manager.health()["state"], "restarting")
        self.assertFalse(self.manager.restart_worker(False, 1))
        release_stop.set()
        self.assertTrue(replacement_started.wait(timeout=1))
        restart.join(timeout=1)

        self.assertEqual(result, [False])
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(self.manager._generation, 2)

    def test_failed_load_is_unavailable(self):
        self.manager._generation = 4
        self.manager._worker_status[self.manager.WORKER_ID] = {
            "worker_id": self.manager.WORKER_ID,
            "generation": 4,
            "state": "loading",
        }
        self.manager._handle({
            "type": "worker_load_error", "worker_id": self.manager.WORKER_ID,
            "generation": 4, "pid": 123, "error": "load failed",
        })

        health = self.manager.health()
        self.assertEqual(health["state"], "unavailable")
        self.assertFalse(health["ready"])

    def test_health_exposes_ready_worker_model_metadata_for_fresh_acceptance(self):
        self.manager._generation = 4
        self.manager._worker_status[self.manager.WORKER_ID] = {
            "worker_id": self.manager.WORKER_ID,
            "generation": 4,
            "state": "ready",
            "metadata": {
                "device_count": 8,
                "model_class": "Gemma4CausalLM",
                "backbone_class": "Gemma4Backbone",
                "dtype": "bfloat16",
                "strict_weight_loading": True,
                "skip_mismatch": False,
                "layout_profile": "gemma4_31b_dense_candidate_a_v1",
                "checkpoint_load_strategy": "keras_hub_native_preset_loader",
                "candidate_a_verified": True,
            },
        }

        runtime = self.manager.health()["runtime"]

        self.assertEqual(runtime["model_class"], "Gemma4CausalLM")
        self.assertEqual(runtime["backbone_class"], "Gemma4Backbone")
        self.assertTrue(runtime["candidate_a_verified"])

    def test_restart_joins_old_worker_before_new_start(self):
        self.install_existing_worker()
        self.install_start_fake()

        self.assertTrue(self.manager.restart_worker(False, 1))

        self.assertLess(
            self.events.index(("join", "old")),
            self.events.index(("start", "worker-2")),
        )

    def test_restart_joins_dead_old_worker_before_new_start(self):
        process = self.install_existing_worker()
        process.alive = False
        self.install_start_fake()

        self.assertTrue(self.manager.restart_worker(False, 1))

        self.assertIn(("join", "old"), self.events)
        self.assertLess(
            self.events.index(("join", "old")),
            self.events.index(("start", "worker-2")),
        )

    def test_shutdown_joins_worker(self):
        self.install_existing_worker()

        self.assertTrue(self.manager.shutdown(False, 1))

        self.assertIn(("join", "old"), self.events)

    def test_shutdown_is_idempotent(self):
        self.install_existing_worker()

        self.assertTrue(self.manager.shutdown(False, 1))
        self.assertTrue(self.manager.shutdown(False, 1))

        self.assertEqual(self.events.count(("join", "old")), 1)

    def test_shutdown_does_not_spawn_replacement(self):
        process = self.install_existing_worker()
        self.install_start_fake()

        self.assertTrue(self.manager.shutdown(False, 1))
        self.manager._monitor(process, 1)

        self.assertEqual(self.starts, [])
        self.assertEqual(self.manager._automatic_restarts_used, 0)


if __name__ == "__main__":
    unittest.main()
