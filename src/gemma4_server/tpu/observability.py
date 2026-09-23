from __future__ import annotations

import logging
import re
from typing import Any


_COMPILE_SECONDS_RE = re.compile(
    r"(?:in|took)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b",
    re.IGNORECASE,
)

_COMPILE_OP_RE = re.compile(
    r"Compiling\s+jit\((\w+)\)",
    re.IGNORECASE,
)

_CACHE_HIT_OP_RE = re.compile(
    r"Persistent\s+(?:compilation\s+)?cache\s+hit\s+for\s+'(\w+)'",
    re.IGNORECASE,
)

_CACHE_MISS_OP_RE = re.compile(
    r"Persistent\s+(?:compilation\s+)?cache\s+miss\s+for\s+'(\w+)'",
    re.IGNORECASE,
)


def parse_operation_identity(message: str) -> str | None:
    m = _COMPILE_OP_RE.search(message)
    if m:
        return "jit_" + m.group(1)
    m = _CACHE_HIT_OP_RE.search(message)
    if m:
        return m.group(1)
    m = _CACHE_MISS_OP_RE.search(message)
    if m:
        return m.group(1)
    return None


def correlate_compile_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    pending_by_op: dict[str, list[str]] = {}
    compile_attempt_count = 0
    unparseable_compile_attempt_count = 0
    matched_hits = 0
    matched_misses = 0
    unmatched_hits = 0
    unmatched_misses = 0
    effective_compile = 0

    for event in events:
        message = str(event.get("message", ""))
        op = parse_operation_identity(message)
        is_compile = bool(event.get("compile"))
        is_hit = bool(event.get("persistent_cache_hit"))
        is_miss = bool(event.get("persistent_cache_miss"))

        if is_compile:
            compile_attempt_count += 1
            if op is None:
                unparseable_compile_attempt_count += 1
            else:
                pending_by_op.setdefault(op, []).append("pending")

        if is_hit:
            pending = pending_by_op.get(op) if op is not None else None
            if pending:
                pending.pop(0)
                matched_hits += 1
            else:
                unmatched_hits += 1

        if is_miss:
            pending = pending_by_op.get(op) if op is not None else None
            if pending:
                pending.pop(0)
                matched_misses += 1
            else:
                unmatched_misses += 1
            effective_compile += 1

    unresolved = unparseable_compile_attempt_count + sum(
        len(pending) for pending in pending_by_op.values()
    )
    persistent_cache_misses = matched_misses + unmatched_misses
    hot_eligible = (
        persistent_cache_misses == 0 and unresolved == 0 and effective_compile == 0
    )

    return {
        "compile_attempt_count": compile_attempt_count,
        "persistent_cache_hits": matched_hits + unmatched_hits,
        "persistent_cache_misses": persistent_cache_misses,
        "matched_cache_hit_count": matched_hits,
        "matched_cache_miss_count": matched_misses,
        "unmatched_cache_hit_count": unmatched_hits,
        "unmatched_cache_miss_count": unmatched_misses,
        "unparseable_compile_attempt_count": unparseable_compile_attempt_count,
        "unresolved_compile_attempt_count": unresolved,
        "effective_compile_count": effective_compile,
        "effective_compile_seconds": 0.0 if hot_eligible else None,
        "hot_eligible": hot_eligible,
    }


def enable_jax_compile_logging() -> dict[str, Any]:
    try:
        import jax

        jax.config.update("jax_log_compiles", True)
        enabled = bool(jax.config.jax_log_compiles)
        return {
            "enabled": enabled,
            "status": "direct" if enabled else "unavailable",
            "error": None,
        }
    except Exception as exc:
        return {
            "enabled": False,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _covered_logger_names(names: tuple[str, ...]) -> tuple[str, ...]:
    unique = sorted(set(names), key=lambda value: (value.count("."), value))
    selected: list[str] = []
    for name in unique:
        if any(name == parent or name.startswith(parent + ".") for parent in selected):
            continue
        selected.append(name)
    return tuple(selected)


class _CompilationHandler(logging.Handler):
    _g9_observer = True

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        lowered = message.lower()
        is_compile = "compiling " in lowered or "compile" in lowered and "jit" in lowered
        is_cache_hit = "persistent compilation cache hit" in lowered or (
            "persistent cache hit" in lowered
        )
        is_cache_miss = "persistent compilation cache miss" in lowered or (
            "persistent cache miss" in lowered
        )
        if not (is_compile or is_cache_hit or is_cache_miss):
            return
        duration_match = _COMPILE_SECONDS_RE.search(message)
        duration = float(duration_match.group(1)) if duration_match else None
        self.events.append(
            {
                "logger": record.name,
                "level": record.levelname,
                "message": message[:500],
                "compile": bool(is_compile),
                "persistent_cache_hit": bool(is_cache_hit),
                "persistent_cache_miss": bool(is_cache_miss),
                "compile_seconds": duration,
            }
        )


class CompilationEvidenceCapture:
    """Capture direct JAX compile/cache log events for one operation."""

    def __init__(
        self,
        logger_names: tuple[str, ...] = ("jax", "jax._src"),
    ):
        self.logger_names = _covered_logger_names(logger_names)
        self._handler: _CompilationHandler | None = None
        self._attached: list[logging.Logger] = []
        self._status = "not_started"
        self._error: str | None = None
        self._compile_logging_enabled = False
        self._coverage_verified = False

    def __enter__(self) -> "CompilationEvidenceCapture":
        self._handler = _CompilationHandler()
        try:
            logging_state = enable_jax_compile_logging()
            self._compile_logging_enabled = bool(logging_state.get("enabled"))
            self._error = logging_state.get("error")
            for name in self.logger_names:
                logger = logging.getLogger(name)
                logger.addHandler(self._handler)
                self._attached.append(logger)
            self._coverage_verified = bool(
                self._attached and self._compile_logging_enabled
            )
            self._status = "direct" if self._coverage_verified else "unavailable"
        except Exception as exc:
            self._status = "unavailable"
            self._error = repr(exc)
            self._remove_handler()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._remove_handler()
        return False

    def _remove_handler(self) -> None:
        handler = self._handler
        for logger in self._attached:
            try:
                logger.removeHandler(handler)
            except (AttributeError, RuntimeError):
                pass
        self._attached.clear()

    def snapshot(self) -> dict[str, Any]:
        events = list(self._handler.events) if self._handler else []
        compile_events = [event for event in events if event["compile"]]
        durations = [
            event["compile_seconds"]
            for event in compile_events
            if event["compile_seconds"] is not None
        ]
        correlation = correlate_compile_events(events)
        return {
            "source": "jax_logging",
            "available": self._status == "direct",
            "events": events,
            "compile_event_count": len(compile_events),
            "compile_seconds": round(sum(durations), 6),
            "persistent_cache_hits": sum(
                1 for event in events if event["persistent_cache_hit"]
            ),
            "persistent_cache_misses": sum(
                1 for event in events if event["persistent_cache_miss"]
            ),
            "status": self._status,
            "compile_logging_enabled": self._compile_logging_enabled,
            "coverage_verified": self._coverage_verified,
            "observer_logger_names": self.logger_names,
            "error": self._error,
            "compile_attempt_count": correlation["compile_attempt_count"],
            "matched_cache_hit_count": correlation["matched_cache_hit_count"],
            "matched_cache_miss_count": correlation["matched_cache_miss_count"],
            "unmatched_cache_hit_count": correlation["unmatched_cache_hit_count"],
            "unmatched_cache_miss_count": correlation["unmatched_cache_miss_count"],
            "unparseable_compile_attempt_count": correlation[
                "unparseable_compile_attempt_count"
            ],
            "unresolved_compile_attempt_count": correlation["unresolved_compile_attempt_count"],
            "effective_compile_count": correlation["effective_compile_count"],
            "effective_compile_seconds": correlation["effective_compile_seconds"],
            "hot_eligible": correlation["hot_eligible"],
        }


def adjudicate_hot_cache(
    prime: dict[str, Any],
    hot1: dict[str, Any],
    hot2: dict[str, Any],
) -> dict[str, Any]:
    captures = (prime, hot1, hot2)
    direct = all(
        capture.get("available") is True
        and capture.get("status") == "direct"
        and capture.get("compile_logging_enabled") is True
        and capture.get("coverage_verified") is True
        for capture in captures
    )

    def _correlation_for(capture: dict[str, Any]) -> dict[str, Any] | None:
        events = capture.get("events")
        if isinstance(events, list) and events:
            return correlate_compile_events(events)
        return None

    def _effective_compile(capture: dict[str, Any]) -> int:
        correlation = _correlation_for(capture)
        if correlation is not None:
            return correlation["effective_compile_count"]
        return capture.get(
            "effective_compile_count", capture.get("compile_event_count", 0)
        )

    def _unresolved(capture: dict[str, Any]) -> int:
        correlation = _correlation_for(capture)
        if correlation is not None:
            return correlation["unresolved_compile_attempt_count"]
        return capture.get("unresolved_compile_attempt_count", 0)

    def _misses(capture: dict[str, Any]) -> int:
        correlation = _correlation_for(capture)
        if correlation is not None:
            return correlation["persistent_cache_misses"]
        return capture.get("persistent_cache_misses", 0)

    hot_zero_effective_compile = all(
        _effective_compile(capture) == 0 for capture in (hot1, hot2)
    )
    hot_no_miss = all(_misses(capture) == 0 for capture in (hot1, hot2))
    hot_no_unresolved = all(
        _unresolved(capture) == 0 for capture in (hot1, hot2)
    )
    passed = direct and hot_zero_effective_compile and hot_no_miss and hot_no_unresolved
    return {
        "hot_cache_reuse": passed,
        "hot_prefill_compile_seconds": 0.0 if passed else None,
        "hot_decode_compile_seconds": 0.0 if passed else None,
        "HOT_CACHE_REUSE": passed,
        "HOT_PREFILL_COMPILE_SECONDS": 0.0 if passed else None,
        "HOT_DECODE_COMPILE_SECONDS": 0.0 if passed else None,
        "compile_evidence": "PASS" if passed else "FAIL",
        "reason": (
            "direct zero-compile evidence for HOT-1 and HOT-2"
            if passed
            else "direct compile/cache evidence was incomplete or showed HOT compilation"
        ),
    }
