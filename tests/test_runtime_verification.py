from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.tpu.authority_contract import (  # noqa: E402
    AuthorityGateError,
    CandidateAVerificationError,
    exact_runtime_versions,
    hardware_gate,
    project_dependency_gate,
    verify_candidate_a,
)


class FakeShard:
    def __init__(self, shape):
        self.data = type("Data", (), {"shape": tuple(shape)})()


class FakeModel:
    def __init__(self, *, shape=(262144, 5376), dtype="bfloat16", count=8, spec_text="P('model','batch')"):
        spec = type("Spec", (), {"__repr__": lambda self: spec_text})()
        sharding = type("Sharding", (), {"spec": spec})()
        value = type(
            "Value",
            (),
            {
                "shape": shape,
                "dtype": dtype,
                "sharding": sharding,
                "addressable_shards": [FakeShard((32768, 5376)) for _ in range(count)],
            },
        )()
        self.weights = [
            type(
                "Weight",
                (),
                {"path": "backbone/token_embedding/embeddings", "value": value},
            )()
        ]


class AuthorityContractTests(unittest.TestCase):
    def test_hardware_gate_rejects_missing_accelerator(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(AuthorityGateError, "accelerator"):
                hardware_gate(td, (), 400 * 1024**3)

    def test_hardware_gate_rejects_small_memory(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(AuthorityGateError, "memory"):
                hardware_gate(td, ("/dev/accel0",), 299 * 1024**3)

    def test_exact_runtime_versions_accepts_only_exact_manifest(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as stream:
            stream.write("keras==3.15.0\nkeras-hub==0.29.1\n")
            stream.flush()
            result = exact_runtime_versions(
                stream.name,
                {"keras": "3.15.0", "keras_hub": "0.29.1"},
            )
        self.assertEqual(result["keras"], "3.15.0")
        self.assertEqual(result["keras-hub"], "0.29.1")

    def test_exact_runtime_versions_rejects_mismatch(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as stream:
            stream.write("keras==3.15.0\n")
            stream.flush()
            with self.assertRaisesRegex(AuthorityGateError, "keras"):
                exact_runtime_versions(stream.name, {"keras": "3.14.0"})

    def test_dependency_gate_ignores_unrelated_conflicts(self):
        distributions = {
            "keras": {"version": "3.15.0", "requires": ["runtime-core>=1"]},
            "runtime-core": {"version": "1.2", "requires": []},
            "unrelated": {"version": "1", "requires": ["missing-package"]},
        }
        result = project_dependency_gate(distributions, ("keras",))
        self.assertTrue(result["pass"])
        self.assertEqual(result["checked_distributions"], ["keras", "runtime-core"])

    def test_dependency_gate_ignores_inactive_extra_marker(self):
        distributions = {
            "root": {
                "version": "1.0",
                "requires": ['ci-only==9; extra == "ci"'],
            },
        }
        result = project_dependency_gate(distributions, ("root",))
        self.assertTrue(result["pass"])

    def test_dependency_gate_accepts_pep440_wildcard_version(self):
        distributions = {
            "root": {"version": "1.0", "requires": ["dep==1.*"]},
            "dep": {"version": "1.0.9", "requires": []},
        }
        result = project_dependency_gate(distributions, ("root",))
        self.assertTrue(result["pass"])

    def test_candidate_a_verification_returns_frozen_contract(self):
        result = verify_candidate_a(FakeModel())
        self.assertTrue(result["CANDIDATE_A_SHARDING_VERIFIED"])
        self.assertEqual(result["TOKEN_EMBEDDING_SHAPE"], [262144, 5376])
        self.assertEqual(result["TOKEN_EMBEDDING_DTYPE"], "bfloat16")
        self.assertEqual(result["TOKEN_EMBEDDING_ADDRESSABLE_SHARDS"], 8)
        self.assertEqual(result["TOKEN_EMBEDDING_SHARD_SHAPES"], [[32768, 5376]] * 8)
        self.assertEqual(result["TOKEN_EMBEDDING_SHARDING_SPEC"], "P('model','batch')")

    def test_candidate_a_verification_rejects_wrong_shard_count(self):
        with self.assertRaises(CandidateAVerificationError):
            verify_candidate_a(FakeModel(count=1))

    def test_candidate_a_verification_rejects_reversed_partition_spec(self):
        with self.assertRaises(CandidateAVerificationError):
            verify_candidate_a(FakeModel(spec_text="P('batch','model')"))


if __name__ == "__main__":
    unittest.main()
