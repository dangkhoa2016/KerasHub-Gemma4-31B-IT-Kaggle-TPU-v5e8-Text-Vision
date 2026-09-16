from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma4_server.tpu.observability import (  # noqa: E402
    CompilationEvidenceCapture,
    adjudicate_hot_cache,
    enable_jax_compile_logging,
)


class CompilationEvidenceCaptureTests(unittest.TestCase):
    def test_captures_compile_event_and_removes_handler(self):
        logger = logging.getLogger("test-jax")
        capture = CompilationEvidenceCapture(logger_names=("test-jax",))

        with capture:
            logger.warning(
                "Compiling jit(generate) with global shapes and types ..."
            )

        evidence = capture.snapshot()
        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["compile_event_count"], 1)
        self.assertEqual(evidence["status"], "direct")
        self.assertEqual(
            len([h for h in logger.handlers if getattr(h, "_g9_observer", False)]),
            0,
        )

    def test_captures_persistent_cache_hits_and_misses(self):
        logger = logging.getLogger("test-cache")
        capture = CompilationEvidenceCapture(logger_names=("test-cache",))

        with capture:
            logger.warning("Persistent compilation cache hit for generate")
            logger.warning("Persistent compilation cache miss for decode")

        evidence = capture.snapshot()
        self.assertEqual(evidence["persistent_cache_hits"], 1)
        self.assertEqual(evidence["persistent_cache_misses"], 1)

    def test_handler_is_removed_when_wrapped_code_raises(self):
        logger = logging.getLogger("test-exception")
        capture = CompilationEvidenceCapture(logger_names=("test-exception",))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with capture:
                raise RuntimeError("boom")

        self.assertEqual(
            len(
                [
                    h
                    for h in logger.handlers
                    if getattr(h, "_g9_observer", False)
                ]
            ),
            0,
        )

    @patch(
        "gemma4_server.tpu.observability.enable_jax_compile_logging",
        return_value={"enabled": True, "status": "direct", "error": None},
    )
    def test_empty_capture_is_direct_zero_compile_evidence(self, enable_logging):
        capture = CompilationEvidenceCapture(logger_names=("test-empty",))

        with capture:
            pass

        evidence = capture.snapshot()
        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["compile_event_count"], 0)
        self.assertEqual(evidence["compile_seconds"], 0.0)
        self.assertEqual(evidence["status"], "direct")
        self.assertTrue(evidence["compile_logging_enabled"])
        self.assertTrue(evidence["coverage_verified"])
        enable_logging.assert_called_once()

    @patch(
        "gemma4_server.tpu.observability.enable_jax_compile_logging",
        return_value={
            "enabled": False,
            "status": "unavailable",
            "error": "jax unavailable",
        },
    )
    def test_empty_capture_is_not_authoritative_without_logging_coverage(
        self, _enable_logging
    ):
        capture = CompilationEvidenceCapture(logger_names=("test-no-coverage",))

        with capture:
            pass

        evidence = capture.snapshot()
        self.assertFalse(evidence["available"])
        self.assertFalse(evidence["coverage_verified"])


class HotCacheAdjudicationTests(unittest.TestCase):
    def evidence(self, **overrides):
        value = {
            "available": True,
            "status": "direct",
            "compile_event_count": 0,
            "compile_seconds": 0.0,
            "persistent_cache_hits": 0,
            "persistent_cache_misses": 0,
            "compile_logging_enabled": True,
            "coverage_verified": True,
        }
        value.update(overrides)
        return value

    def test_passes_only_when_both_hot_requests_have_direct_zero_compile(self):
        result = adjudicate_hot_cache(
            self.evidence(compile_event_count=1, compile_seconds=12.5),
            self.evidence(),
            self.evidence(),
        )

        self.assertTrue(result["hot_cache_reuse"])
        self.assertEqual(result["hot_prefill_compile_seconds"], 0.0)
        self.assertEqual(result["hot_decode_compile_seconds"], 0.0)
        self.assertEqual(result["compile_evidence"], "PASS")
        self.assertTrue(result["HOT_CACHE_REUSE"])
        self.assertEqual(result["HOT_PREFILL_COMPILE_SECONDS"], 0.0)
        self.assertEqual(result["HOT_DECODE_COMPILE_SECONDS"], 0.0)

    def test_fails_when_observer_coverage_is_unverified(self):
        result = adjudicate_hot_cache(
            self.evidence(coverage_verified=False),
            self.evidence(),
            self.evidence(),
        )

        self.assertFalse(result["hot_cache_reuse"])
        self.assertIsNone(result["HOT_PREFILL_COMPILE_SECONDS"])

    def test_fails_when_evidence_is_unavailable(self):
        result = adjudicate_hot_cache(
            self.evidence(),
            self.evidence(available=False, status="unavailable"),
            self.evidence(),
        )

        self.assertFalse(result["hot_cache_reuse"])
        self.assertIsNone(result["hot_prefill_compile_seconds"])
        self.assertEqual(result["compile_evidence"], "FAIL")

    def test_fails_when_hot_request_has_compile_event_or_cache_miss(self):
        result = adjudicate_hot_cache(
            self.evidence(),
            self.evidence(compile_event_count=1),
            self.evidence(persistent_cache_misses=1),
        )

        self.assertFalse(result["hot_cache_reuse"])
        self.assertEqual(result["compile_evidence"], "FAIL")


if __name__ == "__main__":
    unittest.main()
