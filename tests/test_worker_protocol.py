from __future__ import annotations

import queue
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.workers.worker import worker_protocol_loop


class FakeMonitor:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class CleanupFailingMonitor(FakeMonitor):
    def stop(self):
        super().stop()
        raise RuntimeError("cleanup failed")


class FakeEngine:
    def generate_text(self, prompt, system, max_tokens):
        return f"{system}: {prompt}", {"max_tokens": max_tokens}


class G8WorkerProtocolTests(unittest.TestCase):
    def events_from(self, result_queue):
        events = []
        while not result_queue.empty():
            events.append(result_queue.get_nowait())
        return events

    def test_successful_fake_load_emits_ready_before_job_lifecycle(self):
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        shutdown_event = threading.Event()
        monitor = FakeMonitor()
        loader_returned = threading.Event()
        loader_calls = []

        def loader():
            loader_calls.append(True)
            loader_returned.set()
            return FakeEngine(), {"device_count": 8, "dtype": "bfloat16"}

        task_queue.put({
            "job_id": "job-1",
            "prompt": "hello",
            "system": "be concise",
            "max_tokens": 12,
        })
        task_queue.put(None)
        thread = threading.Thread(
            target=worker_protocol_loop,
            args=(
                "tpu-0", 3, task_queue, result_queue, shutdown_event,
                loader, monitor,
            ),
        )
        thread.start()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(loader_calls, [True])
        events = self.events_from(result_queue)
        self.assertEqual(
            [event["type"] for event in events[:2]],
            ["worker_state", "worker_ready"],
        )
        self.assertTrue(loader_returned.is_set())
        self.assertEqual(events[0]["state"], "loading")
        self.assertEqual(
            events[1]["metadata"], {"device_count": 8, "dtype": "bfloat16"},
        )
        self.assertEqual(
            [event["type"] for event in events[2:4]],
            ["job_started", "job_completed"],
        )

    def test_loader_failure_emits_error_without_ready(self):
        result_queue = queue.Queue()
        monitor = FakeMonitor()
        loader_started = threading.Event()
        release_loader_failure = threading.Event()

        def loader():
            loader_started.set()
            self.assertTrue(release_loader_failure.wait(timeout=1))
            raise RuntimeError("load failed")

        thread = threading.Thread(
            target=worker_protocol_loop,
            args=(
                "tpu-0", 3, queue.Queue(), result_queue, threading.Event(),
                loader, monitor,
            ),
        )
        thread.start()
        self.assertTrue(loader_started.wait(timeout=1))
        before_failure = self.events_from(result_queue)
        self.assertEqual(
            [event["type"] for event in before_failure],
            ["worker_state"],
        )
        release_loader_failure.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        events = before_failure + self.events_from(result_queue)
        self.assertEqual(
            [event["type"] for event in events],
            ["worker_state", "worker_load_error"],
        )
        self.assertEqual(events[-1]["error"], "RuntimeError('load failed')")
        self.assertEqual(monitor.stop_calls, 1)

    def test_compile_failure_inside_loader_emits_load_error_without_ready(self):
        result_queue = queue.Queue()
        monitor = FakeMonitor()
        compile_started = threading.Event()
        allow_compile_failure = threading.Event()

        def loader():
            compile_started.set()
            self.assertTrue(allow_compile_failure.wait(timeout=1))
            raise RuntimeError("compile failed")

        thread = threading.Thread(
            target=worker_protocol_loop,
            args=(
                "tpu-0", 3, queue.Queue(), result_queue, threading.Event(),
                loader, monitor,
            ),
        )
        thread.start()
        self.assertTrue(compile_started.wait(timeout=1))
        self.assertEqual(
            [event["type"] for event in self.events_from(result_queue)],
            ["worker_state"],
        )
        allow_compile_failure.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        events = self.events_from(result_queue)
        self.assertEqual(
            [event["type"] for event in events],
            ["worker_load_error"],
        )
        self.assertEqual(events[0]["error"], "RuntimeError('compile failed')")
        self.assertEqual(monitor.stop_calls, 1)

    def test_generation_failure_emits_job_failed(self):
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        monitor = FakeMonitor()
        generation_started = threading.Event()
        release_generation_failure = threading.Event()
        test_case = self

        class EventGatedFailingEngine:
            def generate_text(self, prompt, system, max_tokens):
                generation_started.set()
                test_case.assertTrue(
                    release_generation_failure.wait(timeout=1)
                )
                raise RuntimeError("generation failed")

        task_queue.put({
            "job_id": "job-2",
            "prompt": "hello",
            "system": "be concise",
            "max_tokens": 12,
        })
        task_queue.put(None)

        thread = threading.Thread(
            target=worker_protocol_loop,
            args=(
                "tpu-0", 3, task_queue, result_queue, threading.Event(),
                lambda: (EventGatedFailingEngine(), {}), monitor,
            ),
        )
        thread.start()
        self.assertTrue(generation_started.wait(timeout=1))
        before_failure = self.events_from(result_queue)
        self.assertEqual(
            [event["type"] for event in before_failure],
            ["worker_state", "worker_ready", "job_started"],
        )
        release_generation_failure.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        events = before_failure + self.events_from(result_queue)
        self.assertEqual(
            [event["type"] for event in events],
            [
                "worker_state", "worker_ready", "job_started", "job_failed",
                "worker_stopped",
            ],
        )
        self.assertEqual(events[3]["job_id"], "job-2")
        self.assertEqual(
            events[3]["error"], "RuntimeError('generation failed')"
        )

    def test_shutdown_is_idempotent(self):
        result_queue = queue.Queue()
        shutdown_event = threading.Event()
        monitor = FakeMonitor()
        loader_calls = []
        loader_started = threading.Event()
        release_loader = threading.Event()

        def loader():
            loader_calls.append(True)
            loader_started.set()
            self.assertTrue(release_loader.wait(timeout=1))
            return FakeEngine(), {}

        thread = threading.Thread(
            target=worker_protocol_loop,
            args=(
                "tpu-0", 3, queue.Queue(), result_queue, shutdown_event,
                loader, monitor,
            ),
        )
        thread.start()
        self.assertTrue(loader_started.wait(timeout=1))
        shutdown_event.set()
        shutdown_event.set()
        release_loader.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        events = self.events_from(result_queue)
        self.assertEqual(loader_calls, [True])
        self.assertEqual(
            [event["type"] for event in events],
            ["worker_state", "worker_ready", "worker_stopped"],
        )
        self.assertEqual(monitor.stop_calls, 1)

    def test_cleanup_failure_follows_completed_job_event(self):
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        monitor = CleanupFailingMonitor()
        cleanup_failure = []
        worker_started = threading.Event()
        release_worker = threading.Event()
        task_queue.put({
            "job_id": "job-cleanup", "prompt": "hello",
            "system": "be concise", "max_tokens": 12,
        })
        task_queue.put(None)

        def run_worker():
            worker_started.set()
            self.assertTrue(release_worker.wait(timeout=1))
            try:
                worker_protocol_loop(
                    "tpu-0", 3, task_queue, result_queue, threading.Event(),
                    lambda: (FakeEngine(), {}), monitor,
                )
            except RuntimeError as exc:
                cleanup_failure.append(exc)

        thread = threading.Thread(target=run_worker)
        thread.start()
        self.assertTrue(worker_started.wait(timeout=1))
        release_worker.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(cleanup_failure), 1)
        self.assertEqual(str(cleanup_failure[0]), "cleanup failed")
        self.assertEqual(monitor.stop_calls, 1)
        self.assertEqual(
            [event["type"] for event in self.events_from(result_queue)],
            ["worker_state", "worker_ready", "job_started", "job_completed"],
        )

    def test_slow_loader_keeps_ready_unemitted(self):
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        monitor = FakeMonitor()
        loader_started = threading.Event()
        release_loader = threading.Event()

        def loader():
            loader_started.set()
            self.assertTrue(release_loader.wait(timeout=1))
            return FakeEngine(), {}

        thread = threading.Thread(
            target=worker_protocol_loop,
            args=(
                "tpu-0", 3, task_queue, result_queue, threading.Event(),
                loader, monitor,
            ),
        )
        thread.start()
        self.assertTrue(loader_started.wait(timeout=1))
        self.assertEqual(
            [event["type"] for event in self.events_from(result_queue)],
            ["worker_state"],
        )

        task_queue.put(None)
        release_loader.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(
            [event["type"] for event in self.events_from(result_queue)],
            ["worker_ready", "worker_stopped"],
        )
