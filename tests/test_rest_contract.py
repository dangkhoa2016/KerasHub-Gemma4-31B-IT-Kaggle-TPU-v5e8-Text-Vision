from __future__ import annotations

import sys
import unittest
import base64
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.core.config import Config
from gemma4_server.core.errors import QueueFullError
from gemma4_server.jobs.models import Job
from gemma4_server.jobs.store import JobStore
from gemma4_server.api.app import Runtime, create_app
from gemma4_server.workers.manager import GenerationManager


class G7TopologyTests(unittest.TestCase):
    def test_start_async_does_not_start_a_second_worker(self):
        manager = GenerationManager(Config.for_tests())
        starts = []

        def fake_start_worker():
            starts.append(True)
            manager._worker_status[manager.WORKER_ID] = {
                "worker_id": manager.WORKER_ID,
                "generation": len(starts),
                "state": "starting",
            }

        manager._start_worker = fake_start_worker
        self.addCleanup(manager._collector_stop.set)

        manager.start_async()
        manager.start_async()

        self.assertEqual(len(starts), 1)

    def test_server_uses_one_waitress_process_and_one_model_worker_identity(self):
        server_source = (ROOT / "src/server.py").read_text(encoding="utf-8")
        app_source = (ROOT / "src/gemma4_server/api/app.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("from waitress import serve", server_source)
        self.assertIn("serve(", server_source)
        self.assertNotIn("gunicorn", server_source.lower())
        self.assertNotIn("--workers", server_source)
        self.assertNotIn("reload=", server_source)
        self.assertNotIn("import jax", app_source)
        self.assertNotIn("Gemma4TPUEngine", app_source)
        self.assertEqual(GenerationManager.WORKER_ID, "tpu-0")


class FakeManager:
    def __init__(self, ready=True):
        self.ready = ready
        self.store = JobStore(20, 60)
        self.submitted = []
        self.submit_error = None

    def health(self):
        state = "ready" if self.ready else "loading"
        return {
            "state": state,
            "ready": self.ready,
            "ready_workers": 1 if self.ready else 0,
            "expected_workers": 1,
            "accepting_jobs": self.ready,
            "worker_generation": 1,
            "automatic_restarts_used": 0,
            "jobs": self.store.stats(),
            "workers": [],
            "runtime": {
                "model": "gemma4_instruct_31b",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": 8,
                "mesh": [1, 8],
            },
        }

    def submit(self, payload, request_id=None):
        self.submitted.append((payload, request_id))
        if self.submit_error is not None:
            raise self.submit_error
        job = Job(
            id=f"job-{len(self.submitted)}",
            prompt=payload["prompt"],
            system=payload.get("system", ""),
            max_tokens=payload["max_tokens"],
            request_id=request_id,
            image=payload.get("image"),
        )
        self.store.put(job)
        self.store.mark_processing(job.id, "fake-tpu-0")
        self.store.mark_completed(
            job.id,
            "fake output",
            0.001,
            {"fake": True},
            {"runtime_validation": "CPU_FAKE"},
        )
        return job


class G7RestContractTests(unittest.TestCase):
    API_HEADERS = {"Authorization": "Bearer test-api-key"}

    def setUp(self):
        self.manager = FakeManager()
        self.app = create_app(
            Runtime(Config.for_tests(), self.manager)
        )
        self.client = self.app.test_client()

    def test_documented_routes_are_registered(self):
        routes = {
            (rule.rule, method)
            for rule in self.app.url_map.iter_rules()
            for method in rule.methods
            if method in {"GET", "POST"}
        }
        for expected in {
            ("/", "GET"),
            ("/health/live", "GET"),
            ("/health/ready", "GET"),
            ("/info", "GET"),
            ("/generate", "POST"),
            ("/generate/async", "POST"),
            ("/generate/image", "POST"),
            ("/generate/image/async", "POST"),
            ("/result/<job_id>", "GET"),
            ("/restart", "POST"),
        }:
            self.assertIn(expected, routes)

    def test_liveness_is_public_and_returns_json(self):
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.get_json()["status"], "alive")

    def test_readiness_reports_unready_manager_without_model_creation(self):
        self.manager.ready = False
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["ready"])
        self.assertEqual(self.manager.submitted, [])

    def test_info_requires_bearer_or_x_api_key_authentication(self):
        self.assertEqual(self.client.get("/info").status_code, 401)
        self.assertEqual(
            self.client.get(
                "/info", headers={"X-API-Key": "test-api-key"}
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get("/info", headers=self.API_HEADERS).status_code,
            200,
        )

    def test_text_sync_and_async_routes_validate_and_submit_payload(self):
        response = self.client.post(
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 2},
            headers={**self.API_HEADERS, "X-Request-ID": "g7-test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["output"], "fake output")
        self.assertEqual(response.headers["X-Request-ID"], "g7-test")
        self.assertEqual(
            self.manager.submitted[0][0]["max_tokens"], 2
        )

        response = self.client.post(
            "/generate/async",
            json={"prompt": "hello", "max_new_tokens": 2},
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 202)
        self.assertIn("result_url", response.get_json())

        response = self.client.post(
            "/generate/async",
            json={
                "prompt": "Hello",
                "system": "",
                "max_new_tokens": 1,
            },
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            self.manager.submitted[-1][0],
            {
                "prompt": "Hello",
                "system": "",
                "max_tokens": 1,
            },
        )

    def test_validation_and_error_responses_are_json(self):
        response = self.client.post(
            "/generate",
            data="{",
            content_type="application/json",
            headers={**self.API_HEADERS, "X-Request-ID": "bad-json"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.get_json()["request_id"], "bad-json")

        response = self.client.post(
            "/generate",
            json={"max_new_tokens": 2},
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

        response = self.client.post(
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 0},
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 400)

    def test_image_json_and_multipart_routes_use_cpu_parser(self):
        image_bytes = io.BytesIO()
        from PIL import Image

        Image.new("RGB", (2, 2), (10, 20, 30)).save(
            image_bytes, format="PNG"
        )
        encoded = base64.b64encode(image_bytes.getvalue()).decode("ascii")

        response = self.client.post(
            "/generate/image",
            json={
                "image_base64": encoded,
                "prompt": "describe",
                "max_new_tokens": 1,
            },
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.submitted[-1][0]["image"].mode, "RGB")

        response = self.client.post(
            "/generate/image/async",
            data={
                "image": (io.BytesIO(image_bytes.getvalue()), "tiny.png"),
                "prompt": "describe",
                "max_new_tokens": "1",
            },
            content_type="multipart/form-data",
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 202)

    def test_result_lookup_and_queue_errors_have_contract_statuses(self):
        missing = self.client.get(
            "/result/missing", headers=self.API_HEADERS
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.content_type, "application/json")

        self.manager.submit_error = QueueFullError("full")
        response = self.client.post(
            "/generate/async",
            json={"prompt": "hello"},
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_json()["error"], "full")

    def test_restart_route_requires_both_api_key_and_restart_secret(self):
        response = self.client.post("/restart", json={})
        self.assertEqual(response.status_code, 401)

        response = self.client.post(
            "/restart",
            json={},
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 401)

    def test_unhandled_server_errors_use_json_error_schema(self):
        self.manager.submit_error = RuntimeError("boom")
        response = self.client.post(
            "/generate/async",
            json={"prompt": "hello"},
            headers=self.API_HEADERS,
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.get_json()["error"], "Internal server error"
        )
