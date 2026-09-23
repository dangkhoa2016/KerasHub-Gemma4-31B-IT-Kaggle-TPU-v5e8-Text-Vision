from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from final_tpu_one_shot import (  # noqa: E402
    adjudicate_g9,
    pre_prime_authority_gate,
    run_g9,
    run_text_acceptance,
    run_vision_acceptance,
    _model_contract,
)


class FakeClient:
    def __init__(self, info_status=200):
        self.calls = []
        self.posts = []
        self.jobs = {}
        self.info_status = info_status

    def get(self, path):
        self.calls.append(("GET", path, None))
        if path == "/health/live":
            return 200, {"status": "alive"}
        if path == "/health/ready":
            return 200, {"ready": True, "state": "ready"}
        if path == "/info":
            return self.info_status, {
                "runtime": {
                    "model": "gemma4_instruct_31b",
                    "backend": "jax",
                    "jax_default_backend": "tpu",
                    "accelerator": "TPU v5e-8",
                    "device_count": 8,
                    "expected_tpu_devices": 8,
                    "mesh": [1, 8],
                    "mesh_shape": [1, 8],
                    "mesh_axis_names": ["batch", "model"],
                    "dtype": "bfloat16",
                    "num_layers": 60,
                    "model_class": "Gemma4CausalLM",
                    "backbone_class": "Gemma4Backbone",
                    "strict_weight_loading": True,
                    "skip_mismatch": False,
                    "layout_profile": "gemma4_31b_dense_candidate_a_v1",
                    "checkpoint_load_strategy": "keras_hub_native_preset_loader",
                    "candidate_a_verified": True,
                    "source_sha": "a" * 40,
                }
            }
        if path == "/":
            return 200, {"service": "Gemma 4 31B Instruct"}
        raise AssertionError(path)

    def post(self, path, payload):
        self.calls.append(("POST", path, payload))
        self.posts.append((path, payload))
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = (path, payload)
        return 202, {"status": "queued", "job_id": job_id}

    def poll(self, job_id):
        path, _payload = self.jobs[job_id]
        if path == "/generate/image/async":
            return {
                "status": "completed",
                "output": "image result",
                "metrics": {
                    "vision_conditioning_present": True,
                    "compile_cache_evidence": {
                        "available": True,
                        "status": "direct",
                    "compile_event_count": 0,
                    "compile_seconds": 0.0,
                    "persistent_cache_hits": 0,
                    "persistent_cache_misses": 0,
                    "compile_logging_enabled": True,
                    "coverage_verified": True,
                },
                },
            }
        is_prime = len(
            [path for path, _payload in self.posts if path == "/generate/async"]
        ) == 1
        return {
            "status": "completed",
            "output": "text result",
            "metrics": {
                "compile_cache_evidence": {
                    "available": True,
                    "status": "direct",
                    "compile_event_count": 1 if is_prime else 0,
                    "compile_seconds": 1.5 if is_prime else 0.0,
                    "persistent_cache_hits": 0,
                    "persistent_cache_misses": 0,
                    "compile_logging_enabled": True,
                    "coverage_verified": True,
                }
            },
        }


class G9OrchestrationTests(unittest.TestCase):
    def test_model_contract_requires_explicit_mesh_authority(self):
        result = _model_contract({
            "model": "gemma4_instruct_31b",
            "backend": "jax",
            "jax_default_backend": "tpu",
            "accelerator": "TPU v5e-8",
            "dtype": "bfloat16",
            "mesh": [1, 8],
            "mesh_axis_names": ["batch", "model"],
            "model_class": "Gemma4CausalLM",
            "backbone_class": "Gemma4Backbone",
            "strict_weight_loading": True,
            "skip_mismatch": False,
            "layout_profile": "gemma4_31b_dense_candidate_a_v1",
            "checkpoint_load_strategy": "keras_hub_native_preset_loader",
            "candidate_a_verified": True,
        })

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["mesh_shape"])

    def test_model_contract_requires_gemma4_layer_count(self):
        result = _model_contract({
            "model": "gemma4_instruct_31b",
            "backend": "jax",
            "jax_default_backend": "tpu",
            "accelerator": "TPU v5e-8",
            "dtype": "bfloat16",
            "mesh": [1, 8],
            "mesh_shape": [1, 8],
            "mesh_axis_names": ["batch", "model"],
            "num_layers": 59,
            "model_class": "Gemma4CausalLM",
            "backbone_class": "Gemma4Backbone",
            "strict_weight_loading": True,
            "skip_mismatch": False,
            "layout_profile": "gemma4_31b_dense_candidate_a_v1",
            "checkpoint_load_strategy": "keras_hub_native_preset_loader",
            "candidate_a_verified": True,
        })

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["num_layers"])

    def test_pre_prime_gate_requires_authenticated_info(self):
        client = FakeClient(info_status=401)

        result = pre_prime_authority_gate(client, "a" * 40)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["/info"])
        self.assertEqual(result["generation_post_count"], 0)
        self.assertEqual([path for path, _payload in client.posts], [])

    def test_pre_prime_gate_rejects_source_sha_mismatch(self):
        client = FakeClient()

        result = pre_prime_authority_gate(client, "b" * 40)

        self.assertFalse(result["passed"])
        self.assertFalse(result["source_sha"]["match"])
        self.assertIn("source SHA", result["failure_reason"])

    def test_run_g9_stops_before_generation_when_info_is_unauthorized(self):
        client = FakeClient(info_status=401)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                client=client,
                evidence_dir=Path(tmp),
                source_identity={
                    "head": "a" * 40,
                    "expected_sha": "a" * 40,
                    "git_sha_exact": True,
                    "worktree_clean": True,
                },
                expected_sha="a" * 40,
                model_reload_count=0,
            )
            result = run_g9(args)
            evidence = json.loads(
                Path(tmp, "00-pre-prime-authority-gate.json").read_text()
            )

        self.assertEqual(result, 1)
        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["PRE_PRIME_AUTHORITY_GATE"], "FAIL")
        self.assertEqual(evidence["GENERATION_POST_COUNT"], 0)
        self.assertEqual([path for path, _payload in client.posts], [])

    def test_adjudication_requires_all_mandatory_rows(self):
        passed = adjudicate_g9(
            {
                "prime": True,
                "hot_1": True,
                "hot_2": True,
                "hot_cache_reuse": True,
                "compile_evidence": "PASS",
                "text_semantic_acceptance": True,
                "vision_semantic_acceptance": True,
                "rest_acceptance": True,
                "oom_delta": 0,
                "PRE_PRIME_AUTHORITY_GATE": "PASS",
                "GENERATION_POST_COUNT": 4,
                "HOT_PREFILL_COMPILE_SECONDS": 0.0,
                "HOT_DECODE_COMPILE_SECONDS": 0.0,
            }
        )
        failed = adjudicate_g9({**passed, "hot_cache_reuse": False})

        self.assertEqual(passed["G9_STATUS"], "CLOSED/PASS")
        self.assertTrue(passed["HOT_CACHE_REUSE"])
        self.assertEqual(passed["HOT_PREFILL_COMPILE_SECONDS"], 0.0)
        self.assertEqual(passed["HOT_DECODE_COMPILE_SECONDS"], 0.0)
        self.assertEqual(failed["G9_STATUS"], "OPEN/FAIL")

    def test_corrective_bucket_adjudication_is_committed(self):
        self.assertTrue(
            (ROOT / "docs" / "G9-CORRECTIVE-BUCKET-ADJUDICATION.md").is_file()
        )

    def test_text_and_vision_use_async_production_routes(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            text = run_text_acceptance(client, Path(tmp), "08-text-semantic")
            vision = run_vision_acceptance(client, Path(tmp), "09-vision-semantic")

        self.assertTrue(text["accepted"])
        self.assertTrue(vision["accepted"])
        self.assertEqual(
            [path for path, _payload in client.posts],
            ["/generate/async", "/generate/image/async"],
        )
        self.assertIn("image_base64", client.posts[1][1])
        base64.b64decode(client.posts[1][1]["image_base64"], validate=True)

    def test_run_g9_uses_one_prime_and_two_identical_hot_requests(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                client=client,
                evidence_dir=Path(tmp),
                source_identity={
                    "head": "a" * 40,
                    "expected_sha": "a" * 40,
                    "git_sha_exact": True,
                    "worktree_clean": True,
                },
                expected_sha="a" * 40,
                model_reload_count=0,
            )
            with patch(
                "final_tpu_one_shot.read_cgroup_snapshot",
                return_value={
                    "memory_current": 0,
                    "memory_max": 1,
                    "memory_events": {
                        "oom": 0,
                        "oom_kill": 0,
                        "oom_group_kill": 0,
                    },
                },
            ):
                result = run_g9(args)

        self.assertEqual(result, 0)
        text_posts = [
            payload
            for path, payload in client.posts
            if path == "/generate/async"
        ]
        self.assertEqual(len(text_posts), 4)
        self.assertEqual(text_posts[0], text_posts[1])
        self.assertEqual(text_posts[1], text_posts[2])
        self.assertFalse(any(path == "/restart" for _method, path, _ in client.calls))


if __name__ == "__main__":
    unittest.main()
