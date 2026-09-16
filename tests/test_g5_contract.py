from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.tpu.g5_vision import (  # noqa: E402
    G5_PROMPT,
    create_synthetic_fixture,
    inspect_vision_preprocess,
    load_fixture_rgb,
    prepare_image_inputs,
    sha256_file,
    validate_non_empty_generation,
)


class FakePreprocessor:
    def __init__(self):
        self.calls = []

    def generate_preprocess(self, inputs, sequence_length=None):
        self.calls.append((inputs, sequence_length))
        return {
            "pixel_values": np.ones((1, 1, 4, 768), dtype=np.float32),
            "pixel_position_ids": np.zeros((1, 1, 4, 2), dtype=np.int32),
            "token_ids": np.ones((1, 12), dtype=np.int32),
            "vision_indices": np.array([[2, 3]], dtype=np.int32),
            "vision_mask": np.array(
                [[False, False, True, True] + [False] * 8], dtype=bool
            ),
            "padding_mask": np.ones((1, 12), dtype=bool),
        }


class G5VisionContractTests(unittest.TestCase):
    def test_synthetic_fixture_is_deterministic_rgb_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            digest = create_synthetic_fixture(path)
            image = load_fixture_rgb(path)

            self.assertEqual(image.shape, (64, 64, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(len(digest), 64)

    def test_prepare_image_inputs_uses_observed_prompt_and_image_structure(self):
        preprocessor = FakePreprocessor()
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        processed, rendered = prepare_image_inputs(
            preprocessor, image, G5_PROMPT, sequence_length=512
        )

        self.assertIn("<|image|>", rendered)
        self.assertEqual(len(preprocessor.calls), 1)
        inputs, length = preprocessor.calls[0]
        self.assertEqual(length, 512)
        self.assertEqual(set(inputs), {"prompts", "images"})
        self.assertEqual(inputs["images"].shape, (64, 64, 3))
        self.assertEqual(processed["token_ids"].shape, (1, 12))

    def test_inspection_requires_real_image_fields_and_vision_positions(self):
        summary = inspect_vision_preprocess(
            {
                "pixel_values": np.ones((1, 1, 4, 768), dtype=np.float32),
                "pixel_position_ids": np.zeros((1, 1, 4, 2), dtype=np.int32),
                "token_ids": np.ones((1, 12), dtype=np.int32),
                "vision_indices": np.array([[2, 3]], dtype=np.int32),
                "vision_mask": np.array(
                    [[False, False, True, True] + [False] * 8], dtype=bool
                ),
                "padding_mask": np.ones((1, 12), dtype=bool),
            }
        )

        self.assertTrue(summary["image_preprocess_returned"])
        self.assertTrue(summary["vision_conditioning_present"])
        self.assertEqual(summary["vision_mask_true_count"], 2)
        self.assertEqual(summary["vision_indices_width"], 2)

    def test_inspection_accepts_installed_unbatched_vision_shapes(self):
        summary = inspect_vision_preprocess(
            {
                "pixel_values": np.ones((1, 2520, 768), dtype=np.float32),
                "pixel_position_ids": np.zeros((1, 2520, 2), dtype=np.int32),
                "token_ids": np.ones((512,), dtype=np.int32),
                "vision_indices": np.arange(280, dtype=np.int32),
                "vision_mask": np.array([True] * 280 + [False] * 232),
                "padding_mask": np.ones((512,), dtype=bool),
            }
        )

        self.assertTrue(summary["image_preprocess_returned"])
        self.assertTrue(summary["vision_conditioning_present"])
        self.assertEqual(summary["vision_indices_width"], 280)

    def test_empty_generation_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_non_empty_generation("  ")

    def test_authority_has_one_load_and_native_vision_order(self):
        source = (ROOT / "scripts/g5_image_authority.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(source.count("Gemma4TPUEngine("), 1)
        self.assertEqual(source.count("engine.load()"), 1)
        self.assertLess(
            source.index("prepare_image_inputs"),
            source.index("engine.generate_image"),
        )
        self.assertLess(
            source.index("verify_candidate_a"),
            source.index("engine.generate_image"),
        )
        self.assertIn('"G5_PATH": "NATIVE_VISION"', source)
        self.assertNotIn("run_g3_tpu_authority", source)
        self.assertNotIn("run_g4_characterization", source)

    def test_authority_records_required_g5_markers(self):
        source = (ROOT / "scripts/g5_image_authority.py").read_text(
            encoding="utf-8"
        )
        for marker in (
            '"IMAGE_INPUT_LOADED"',
            '"IMAGE_PREPROCESS_RETURNED"',
            '"VISION_CONDITIONING_PRESENT"',
            '"G5_GENERATION_STARTED"',
            '"G5_GENERATION_RETURNED"',
            '"G5_GENERATION_CALL_COUNT"',
            "memory.current",
            "memory.events",
            "STRICT_LOAD_HARD_MAX_SECONDS = 2304",
        ):
            self.assertIn(marker, source)

    def test_runner_has_cpu_first_budget_and_required_evidence(self):
        source = (ROOT / "scripts/run_g5_image_authority.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_g3_tpu_authority", source)
        self.assertNotIn("run_g4_characterization", source)
        for marker in (
            "PRIMARY_G5_TPU_ATTEMPTS=1",
            "CORRECTIVE_G5_TPU_RERUNS_ALLOWED=1",
            "MAX_TOTAL_G5_31B_TPU_ATTEMPTS=2",
            "G5_CPU_PREPARATION=PASS",
            "00-g4-freeze-reference.txt",
            "01-g5-source-hashes-before.txt",
            "02-g5-source-hashes-after.txt",
            "03-g5-api-discovery.txt",
            "04-image-fixture.txt",
            "05-hardware-gate.txt",
            "06-runtime.txt",
            "07-dependency-gate.txt",
            "08-memory-before.txt",
            "09-g5-authority.stdout.log",
            "10-g5-authority.stderr.log",
            "11-g5-result.json",
            "12-memory-after.txt",
            "13-final-adjudication.txt",
            "G5_TPU_ATTEMPT_BUDGET_EXHAUSTED",
        ):
            self.assertIn(marker, source)

    def test_source_hash_manifest_is_post_g4_aware(self):
        source = (ROOT / "scripts/run_g5_image_authority.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("POST_G4_SOURCE_CHANGE", source)
        self.assertIn("g4-pass-freeze.json", source)


if __name__ == "__main__":
    unittest.main()
