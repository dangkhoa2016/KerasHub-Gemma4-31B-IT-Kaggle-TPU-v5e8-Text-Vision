from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from final_tpu_one_shot import (  # noqa: E402
    adjudicate_g10,
    session_fresh_from_checkpoint,
    run_g10,
)


class FakeG10Client:
    def __init__(self):
        self.posts = []
        self.jobs = {}
        self.calls = []

    def get(self, path):
        self.calls.append(("GET", path))
        if path == "/info":
            return 200, {
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
                }
            }
        if path == "/health/ready":
            return 200, {"ready": True, "state": "ready"}
        if path == "/health/live":
            return 200, {"status": "alive"}
        if path == "/":
            return 200, {"service": "Gemma 4 31B Instruct"}
        raise AssertionError(path)

    def post(self, path, payload):
        self.calls.append(("POST", path))
        self.posts.append((path, payload))
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = path
        return 202, {"status": "queued", "job_id": job_id}

    def poll(self, job_id):
        path = self.jobs[job_id]
        metrics = {
            "vision_conditioning_present": path == "/generate/image/async",
            "compile_cache_evidence": {
                "available": True,
                "status": "direct",
                "compile_event_count": 0,
                "compile_seconds": 0.0,
                "persistent_cache_hits": 0,
                "persistent_cache_misses": 0,
            },
        }
        return {"status": "completed", "output": "ok", "metrics": metrics}


class G10OrchestrationTests(unittest.TestCase):
    def test_session_fresh_requires_changed_runtime_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "00-context.txt"
            checkpoint.write_text(
                json.dumps(
                    {
                        "runtime_identity_before": {
                            "boot_id": "old-boot",
                            "hostname": "host",
                            "pid1": "old-pid1",
                            "jupyter_parent_pid": "old-jupyter",
                        }
                    }
                ),
                encoding="utf-8",
            )
            # The current container identity is not expected to match this
            # synthetic pre-restart checkpoint.
            fresh, evidence = session_fresh_from_checkpoint(checkpoint)

        self.assertTrue(fresh)
        self.assertTrue(evidence["changed_fields"])

    def test_adjudication_requires_fresh_sha_model_acceptance_and_oom(self):
        passed = adjudicate_g10(
            {
                "session_fresh": True,
                "git_sha_exact": True,
                "tpu_device_count": 8,
                "model_load": True,
                "text_acceptance": True,
                "vision_acceptance": True,
                "rest_lifecycle_acceptance": True,
                "oom_delta": 0,
                "evidence_packaged": True,
            }
        )
        failed = adjudicate_g10({**passed, "session_fresh": False})

        self.assertEqual(passed["G10_STATUS"], "CLOSED/PASS")
        self.assertEqual(failed["G10_STATUS"], "OPEN/FAIL")

    def test_run_g10_passes_complete_fresh_contract(self):
        client = FakeG10Client()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                client=client,
                evidence_dir=Path(tmp),
                session_fresh=True,
                source_identity={
                    "head": "b" * 40,
                    "expected_sha": "b" * 40,
                    "git_sha_exact": True,
                    "worktree_clean": True,
                },
                expected_sha="b" * 40,
                model_reload_count=1,
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
                result = run_g10(args)

        self.assertEqual(result, 0)
        self.assertTrue(any(path == "/info" for _method, path in client.calls))
        self.assertFalse(any(path == "/restart" for _method, path in client.calls))

    def test_run_g10_fails_when_session_is_not_fresh(self):
        client = FakeG10Client()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                client=client,
                evidence_dir=Path(tmp),
                session_fresh=False,
                source_identity={
                    "head": "c" * 40,
                    "expected_sha": "c" * 40,
                    "git_sha_exact": True,
                    "worktree_clean": True,
                },
                expected_sha="c" * 40,
            )
            result = run_g10(args)

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
