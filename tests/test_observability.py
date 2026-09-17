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
    correlate_compile_events,
    enable_jax_compile_logging,
    parse_operation_identity,
)


class CompilationEvidenceCaptureTests(unittest.TestCase):
    @patch(
        "gemma4_server.tpu.observability.enable_jax_compile_logging",
        return_value={"enabled": True, "status": "direct", "error": None},
    )
    def test_captures_compile_event_and_removes_handler(self, _enable_logging):
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


class OperationIdentityParseTests(unittest.TestCase):
    """T10 — malformed/unparseable operation identity."""

    def test_parse_operation_identity_from_compiling_jit_while(self):
        op = parse_operation_identity(
            "Compiling jit(while) with global shapes and types ..."
        )
        self.assertEqual(op, "jit_while")

    def test_parse_operation_identity_from_cache_hit(self):
        op = parse_operation_identity(
            "Persistent compilation cache hit for 'jit_while' with key 'k'"
        )
        self.assertEqual(op, "jit_while")

    def test_parse_operation_identity_from_cache_miss(self):
        op = parse_operation_identity(
            "Persistent compilation cache miss for 'jit_while'"
        )
        self.assertEqual(op, "jit_while")

    def test_parse_operation_identity_returns_none_for_unparseable(self):
        op = parse_operation_identity("some unrelated log message")
        self.assertIsNone(op)


class CorrelationTests(unittest.TestCase):
    """T1-T6: compile/cache correlation tests."""

    def _make_compile_event(self, message, logger="jax._src.interpreters.pxla"):
        return {
            "logger": logger,
            "level": "WARNING",
            "message": message,
            "compile": True,
            "persistent_cache_hit": False,
            "persistent_cache_miss": False,
            "compile_seconds": None,
        }

    def _make_cache_hit_event(
        self, message, logger="jax._src.compiler"
    ):
        return {
            "logger": logger,
            "level": "WARNING",
            "message": message,
            "compile": False,
            "persistent_cache_hit": True,
            "persistent_cache_miss": False,
            "compile_seconds": None,
        }

    def _make_cache_miss_event(
        self, message, logger="jax._src.compiler"
    ):
        return {
            "logger": logger,
            "level": "WARNING",
            "message": message,
            "compile": False,
            "persistent_cache_hit": False,
            "persistent_cache_miss": True,
            "compile_seconds": None,
        }

    def test_t1_matching_hit_resolves_compile(self):
        """T1: compile + matching cache hit -> effective_compile_count=0."""
        events = [
            self._make_compile_event(
                "Compiling jit(while) with global shapes and types ..."
            ),
            self._make_cache_hit_event(
                "Persistent compilation cache hit for 'jit_while' with key 'k'"
            ),
        ]
        result = correlate_compile_events(events)
        self.assertEqual(result["compile_attempt_count"], 1)
        self.assertEqual(result["persistent_cache_hits"], 1)
        self.assertEqual(result["persistent_cache_misses"], 0)
        self.assertEqual(result["matched_cache_hit_count"], 1)
        self.assertEqual(result["unresolved_compile_attempt_count"], 0)
        self.assertEqual(result["effective_compile_count"], 0)
        self.assertEqual(result["effective_compile_seconds"], 0.0)

    def test_t2_matching_miss_resolves_compile(self):
        """T2: compile + matching cache miss -> HOT eligible=false."""
        events = [
            self._make_compile_event(
                "Compiling jit(while) with global shapes and types ..."
            ),
            self._make_cache_miss_event(
                "Persistent compilation cache miss for 'jit_while'"
            ),
        ]
        result = correlate_compile_events(events)
        self.assertEqual(result["matched_cache_miss_count"], 1)
        self.assertEqual(result["effective_compile_count"], 1)
        self.assertFalse(result["hot_eligible"])

    def test_t3_unresolved_compile(self):
        """T3: compile only, no hit/miss -> unresolved."""


        events = [
            self._make_compile_event(
                "Compiling jit(while) with global shapes and types ..."
            ),
        ]
        result = correlate_compile_events(events)
        self.assertEqual(result["unresolved_compile_attempt_count"], 1)
        self.assertFalse(result["hot_eligible"])

    def test_t4_mismatched_cache_hit(self):
        """T4: compile jit_while + cache hit jit_other -> unresolved."""


        events = [
            self._make_compile_event(
                "Compiling jit(while) with global shapes and types ..."
            ),
            self._make_cache_hit_event(
                "Persistent compilation cache hit for 'jit_other' with key 'k'"
            ),
        ]
        result = correlate_compile_events(events)
        self.assertEqual(result["unresolved_compile_attempt_count"], 1)
        self.assertFalse(result["hot_eligible"])

    def test_t5_multiple_fully_matched_hits(self):
        """T5: two compiles each matched by its own cache hit -> HOT eligible."""


        events = [
            self._make_compile_event(
                "Compiling jit(while) with global shapes and types ..."
            ),
            self._make_cache_hit_event(
                "Persistent compilation cache hit for 'jit_while' with key 'k1'"
            ),
            self._make_compile_event(
                "Compiling jit(generate) with global shapes and types ..."
            ),
            self._make_cache_hit_event(
                "Persistent compilation cache hit for 'jit_generate' with key 'k2'"
            ),
        ]
        result = correlate_compile_events(events)
        self.assertEqual(result["effective_compile_count"], 0)
        self.assertEqual(result["unresolved_compile_attempt_count"], 0)
        self.assertTrue(result["hot_eligible"])

    def test_t6_mixed_hits_plus_one_miss(self):
        """T6: one matched hit + one matched miss -> HOT eligible=false."""


        events = [
            self._make_compile_event(
                "Compiling jit(while) with global shapes and types ..."
            ),
            self._make_cache_hit_event(
                "Persistent compilation cache hit for 'jit_while' with key 'k1'"
            ),
            self._make_compile_event(
                "Compiling jit(generate) with global shapes and types ..."
            ),
            self._make_cache_miss_event(
                "Persistent compilation cache miss for 'jit_generate'"
            ),
        ]
        result = correlate_compile_events(events)
        self.assertFalse(result["hot_eligible"])


class CorrelationAdjudicationIntegrationTests(unittest.TestCase):
    """T7-T9: adjudication integration with correlation."""

    def evidence(self, events=None, **overrides):
        value = {
            "available": True,
            "status": "direct",
            "events": events or [],
            "compile_event_count": 0,
            "compile_seconds": 0.0,
            "persistent_cache_hits": 0,
            "persistent_cache_misses": 0,
            "compile_logging_enabled": True,
            "coverage_verified": True,
        }
        value.update(overrides)
        return value

    def test_t7_observer_coverage_unverified(self):
        """T7: unverified coverage -> HOT eligible=false."""
        result = adjudicate_hot_cache(
            self.evidence(coverage_verified=False),
            self.evidence(),
            self.evidence(),
        )
        self.assertFalse(result["hot_cache_reuse"])

    def test_t8_direct_verified_empty_capture(self):
        """T8: verified empty capture -> compile_attempt_count=0, HOT eligible."""
        result = adjudicate_hot_cache(
            self.evidence(),
            self.evidence(),
            self.evidence(),
        )
        self.assertTrue(result["hot_cache_reuse"])
        self.assertEqual(result["hot_prefill_compile_seconds"], 0.0)

    def test_t9_historical_jax_0_10_2_hot_fixture(self):
        """T9: historical JAX 0.10.2 pattern recognized as cache-hit-resolved."""
        hot_events = [
            {
                "logger": "jax._src.interpreters.pxla",
                "level": "WARNING",
                "message": (
                    "Compiling jit(while) with global shapes and types "
                    "(ShapedArray(int32[]), ...)"
                ),
                "compile": True,
                "persistent_cache_hit": False,
                "persistent_cache_miss": False,
                "compile_seconds": None,
            },
            {
                "logger": "jax._src.compiler",
                "level": "WARNING",
                "message": (
                    "Persistent compilation cache hit for 'jit_while' "
                    "with key 'jit_while-51a1c7f33eff15b62b96e52065037d22'"
                ),
                "compile": False,
                "persistent_cache_hit": True,
                "persistent_cache_miss": False,
                "compile_seconds": None,
            },
        ]
        prime = self.evidence(compile_event_count=1)
        hot1 = self.evidence(events=hot_events, compile_event_count=1)
        hot2 = self.evidence(events=list(hot_events), compile_event_count=1)
        result = adjudicate_hot_cache(prime, hot1, hot2)
        self.assertTrue(result["hot_cache_reuse"], result.get("reason"))
        self.assertEqual(result["hot_prefill_compile_seconds"], 0.0)
        self.assertEqual(result["hot_decode_compile_seconds"], 0.0)

    def test_t9_historical_fixture_re_adjudication(self):
        """Historical fixture unit-level test: cache-hit-resolved HOT pattern."""
        hot_events = [
            {
                "logger": "jax._src.interpreters.pxla",
                "level": "WARNING",
                "message": (
                    "Compiling jit(while) with global shapes and types "
                    "(ShapedArray(int32[]), ShapedArray(bfloat16[262144,5376]))"
                ),
                "compile": True,
                "persistent_cache_hit": False,
                "persistent_cache_miss": False,
                "compile_seconds": None,
            },
            {
                "logger": "jax._src.compiler",
                "level": "WARNING",
                "message": (
                    "Persistent compilation cache hit for 'jit_while' "
                    "with key 'jit_while-51a1c7f33eff15b62b96e52065037d22518e82cb'"
                ),
                "compile": False,
                "persistent_cache_hit": True,
                "persistent_cache_miss": False,
                "compile_seconds": None,
            },
        ]
        prime = self.evidence(compile_event_count=1)
        hot1 = self.evidence(events=hot_events, compile_event_count=1)
        hot2 = self.evidence(events=list(hot_events), compile_event_count=1)
        result = adjudicate_hot_cache(prime, hot1, hot2)
        self.assertTrue(result["HOT_CACHE_REUSE"], result.get("reason"))
        self.assertEqual(result["HOT_PREFILL_COMPILE_SECONDS"], 0.0)
        self.assertEqual(result["HOT_DECODE_COMPILE_SECONDS"], 0.0)
        self.assertEqual(result["compile_evidence"], "PASS")


if __name__ == "__main__":
    unittest.main()
