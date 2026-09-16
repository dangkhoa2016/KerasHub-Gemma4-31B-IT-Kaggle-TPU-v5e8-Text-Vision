from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.tpu.engine import (  # noqa: E402
    Gemma4TPUEngine,
    post_load_host_cleanup,
)


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return " generated text "


class GenerationArchitectureCorrectiveTests(unittest.TestCase):
    def test_engine_compiles_generation_without_jax_jit_peak(self):
        text = (ROOT / "src/gemma4_server/tpu/engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('self.model.compile(sampler="greedy", run_eagerly=True)', text)

    def test_authority_generation_makes_one_exact_native_call(self):
        engine = object.__new__(Gemma4TPUEngine)
        engine.model = FakeModel()
        engine.preprocessor = object()
        engine.max_generation_length = 2048
        engine.buckets = (512, 768, 1024, 1536, 2048)
        engine._preprocess_prompt_tokens = mock.Mock(return_value=10)

        output, metadata = engine.generate_text_authority("Hello", "", 1)

        self.assertEqual(output, "generated text")
        self.assertEqual(len(engine.model.calls), 1)
        _, kwargs = engine.model.calls[0]
        self.assertEqual(kwargs["max_length"], 11)
        self.assertTrue(kwargs["strip_prompt"])
        self.assertNotIn("max_new_tokens", kwargs)
        self.assertEqual(metadata["max_new_tokens"], 1)
        self.assertEqual(metadata["authority_max_length"], 11)
        self.assertEqual(
            metadata["authority_generation_path"],
            "EXACT_LENGTH_AUTHORITY_PATH",
        )

    def test_post_load_cleanup_is_non_fatal_when_trim_is_unavailable(self):
        with mock.patch("gemma4_server.tpu.engine.gc.collect", return_value=7):
            with mock.patch(
                "gemma4_server.tpu.engine.ctypes.CDLL",
                side_effect=OSError("libc unavailable"),
            ):
                result = post_load_host_cleanup()

        self.assertEqual(result["post_load_gc_collected_objects"], 7)
        self.assertFalse(result["post_load_malloc_trim_available"])
        self.assertIsNone(result["post_load_malloc_trim_result"])

    def test_cleanup_records_rss_and_cgroup_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            cgroup = Path(td)
            (cgroup / "memory.current").write_text("123\n", encoding="utf-8")
            with mock.patch(
                "gemma4_server.tpu.engine._CGROUP_MEMORY_CURRENT_PATH",
                cgroup / "memory.current",
            ):
                with mock.patch(
                    "gemma4_server.tpu.engine._read_rss_kib",
                    side_effect=[100, 90, 80],
                ):
                    with mock.patch(
                        "gemma4_server.tpu.engine.ctypes.CDLL",
                        side_effect=OSError("libc unavailable"),
                    ):
                        result = post_load_host_cleanup()

        self.assertEqual(result["post_load_rss_before_cleanup_kib"], 100)
        self.assertEqual(result["post_load_rss_after_gc_kib"], 90)
        self.assertEqual(result["post_load_rss_after_malloc_trim_kib"], 80)
        self.assertEqual(result["cgroup_memory_current_before_cleanup"], 123)


if __name__ == "__main__":
    unittest.main()
