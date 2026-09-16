from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.tpu.g4_split import (  # noqa: E402
    G4_MAX_LENGTH,
    G4_PROMPT,
    G4_REQUESTED_NEW_TOKENS,
    rendered_authority_prompt,
    run_split_generation,
)
from scripts.g4_evidence import native_baseline_from_archive  # noqa: E402


class FakePreprocessor:
    def __init__(self):
        self.calls = []

    def generate_preprocess(self, prompts, sequence_length=None):
        self.calls.append((prompts, sequence_length))
        return {
            "token_ids": np.arange(11, dtype=np.int32)[None, :],
            "padding_mask": np.array(
                [[True] * 10 + [False]], dtype=bool
            ),
        }

    def generate_postprocess(self, result):
        self.postprocess_input = result
        return "generated"


class FakeModel:
    def __init__(self):
        self.preprocessor = FakePreprocessor()
        self.build_cache_calls = []
        self.cache_calls = []

    def generate(self, *args, **kwargs):
        raise AssertionError("G4 split must not call native generate()")

    def _build_cache(self, **kwargs):
        self.build_cache_calls.append(kwargs)
        return np.zeros((1, 11, 4), dtype=np.float32), "prefilled-cache"

    def call_with_cache(self, **kwargs):
        self.cache_calls.append(kwargs)
        logits = np.zeros((1, 1, 8), dtype=np.float32)
        logits[0, 0, 3] = 10.0
        return logits, np.zeros((1, 1, 4), dtype=np.float32), "decoded-cache"


class G4SplitContractTests(unittest.TestCase):
    def test_split_uses_exact_authority_prompt_and_length(self):
        model = FakeModel()

        result = run_split_generation(model, G4_PROMPT)

        self.assertEqual(result["prompt_text"], "Hello")
        self.assertEqual(result["requested_new_tokens"], 1)
        self.assertEqual(result["max_length"], G4_MAX_LENGTH)
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["result"], "generated")
        self.assertEqual(
            model.preprocessor.calls,
            [([rendered_authority_prompt(G4_PROMPT)], G4_MAX_LENGTH)],
        )

    def test_split_prefills_then_decodes_without_native_generate(self):
        model = FakeModel()

        result = run_split_generation(model, G4_PROMPT)

        self.assertEqual(result["generation_call_count"], 1)
        self.assertEqual(len(model.build_cache_calls), 1)
        self.assertEqual(len(model.cache_calls), 1)
        self.assertEqual(model.build_cache_calls[0]["img_embeddings"], None)
        self.assertEqual(model.build_cache_calls[0]["vision_mask"], None)
        decode = model.cache_calls[0]
        self.assertEqual(decode["cache"], "prefilled-cache")
        self.assertEqual(decode["cache_update_index"], 9)
        np.testing.assert_array_equal(
            decode["token_ids"], np.array([[9]], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            decode["cache_update_mask"], np.array([[False]], dtype=bool)
        )

    def test_split_source_has_no_native_generate_call(self):
        source = (ROOT / "src/gemma4_server/tpu/g4_split.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("model.generate(", source)
        self.assertIn("_build_cache", source)
        self.assertIn("call_with_cache", source)

    def test_authority_is_one_process_and_candidate_precedes_split(self):
        source = (ROOT / "scripts/g4_split_authority.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(source.count("Gemma4TPUEngine("), 1)
        self.assertEqual(source.count("engine.load()"), 1)
        self.assertLess(
            source.index("verify_candidate_a"),
            source.index("run_split_generation"),
        )
        self.assertNotIn("bash scripts/run_g3_tpu_authority.sh", source)
        self.assertNotIn("python3 scripts/run_g3_tpu_authority.sh", source)

    def test_authority_has_load_and_split_contract_markers(self):
        source = (ROOT / "scripts/g4_split_authority.py").read_text(
            encoding="utf-8"
        )
        for marker in (
            '"MODEL_LOAD_STARTED"',
            '"MODEL_LOAD_RETURNED"',
            '"G4_SPLIT_PATH_STARTED"',
            '"G4_SPLIT_PATH_RETURNED"',
            '"G4_SPLIT_GENERATION_CALL_COUNT"',
            "memory.current",
            "memory.events",
            "STRICT_LOAD_HARD_MAX_SECONDS = 2304",
        ):
            self.assertIn(marker, source)

    def test_authority_starts_init_watchdog_before_jax_and_cancels_safely(self):
        source = (ROOT / "scripts/g4_split_authority.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            source.index("init_watchdog.start()"),
            source.index("import jax"),
        )
        self.assertIn("if self.thread.is_alive()", source)

    def test_native_baseline_reads_frozen_g3_archive(self):
        archive = ROOT / "artifacts/g3/gemma4-31b-vnext-g3-authority-20260913T225039Z.tar.gz"
        self.assertTrue(archive.is_file())

        result = native_baseline_from_archive(archive)

        self.assertEqual(result["source"], "frozen-g3-authority")
        self.assertEqual(result["prompt_text"], "Hello")
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["requested_new_tokens"], 1)
        self.assertEqual(result["authority_max_length"], 11)
        self.assertEqual(result["generation_seconds"], 567.709641)
        self.assertEqual(result["generation_call_count"], 1)
        self.assertEqual(result["tpu_device_count"], 8)
        self.assertTrue(result["candidate_a_sharding_verified"])
        self.assertTrue(result["run_eagerly"])

    def test_evidence_parser_does_not_rerun_g3(self):
        source = (ROOT / "scripts/g4_evidence.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_g3_tpu_authority", source)
        self.assertNotIn("subprocess", source)

    def test_g4_runner_has_pre_tpu_gates_and_both_exposure_styles(self):
        source = (ROOT / "scripts/run_g4_characterization.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("bash scripts/run_g3_tpu_authority.sh", source)
        self.assertNotIn("python3 scripts/run_g3_tpu_authority.sh", source)
        self.assertIn("vfio_devices", source)
        self.assertIn("/dev/vfio/[0-9]*", source)
        self.assertIn("TPU_DEVICE_EXPOSURE_STYLE", source)
        self.assertIn("configure_kaggle_tpu.sh", source)
        self.assertIn("requirements-tpu-g3.txt", source)
        hardware = source.index("/dev/accel")
        runtime = source.index("exact_runtime_versions")
        authority = source.index(
            "python3 scripts/g4_split_authority.py --evidence-dir"
        )
        self.assertLess(hardware, runtime)
        self.assertLess(runtime, authority)

    def test_g4_runner_has_low_memory_and_no_tpu_evidence_path(self):
        source = (ROOT / "scripts/run_g4_characterization.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("MINIMUM_AUTHORITY_MEMORY_GIB=300", source)
        self.assertIn('"AUTHORITY_NOT_STARTED": True', source)
        self.assertIn("TPU_HARDWARE_NOT_READY", source)
        self.assertIn("exit 1", source)

    def test_g4_runner_has_required_evidence_package_and_budget_markers(self):
        source = (ROOT / "scripts/run_g4_characterization.sh").read_text(
            encoding="utf-8"
        )
        for name in (
            "00-g3-freeze-reference.txt",
            "01-g4-source-hashes-before.txt",
            "02-g4-source-hashes-after.txt",
            "03-api-discovery.txt",
            "04-hardware-gate.txt",
            "05-runtime.txt",
            "06-dependency-gate.txt",
            "07-g4-authority.stdout.log",
            "08-g4-authority.stderr.log",
            "09-g4-result.json",
            "10-memory-before.txt",
            "11-memory-after.txt",
            "12-native-vs-split-comparison.json",
            "13-final-adjudication.txt",
            "SHA256SUMS",
            "G4_TPU_ATTEMPT_BUDGET_EXHAUSTED",
        ):
            self.assertIn(name, source)
        self.assertIn("tar -czf", source)
        self.assertIn("sha256sum \"$archive\"", source)

    def test_g4_runner_copies_authority_result_into_required_package_slot(self):
        source = (ROOT / "scripts/run_g4_characterization.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'cp "$evidence_dir/07-g4-result.json" "$evidence_dir/09-g4-result.json"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
