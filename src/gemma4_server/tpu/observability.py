from __future__ import annotations

import logging
import re
from typing import Any


_COMPILE_SECONDS_RE = re.compile(
    r"(?:in|took)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b",
    re.IGNORECASE,
)


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

    def __enter__(self) -> "CompilationEvidenceCapture":
        self._handler = _CompilationHandler()
        try:
            for name in self.logger_names:
                logger = logging.getLogger(name)
                logger.addHandler(self._handler)
                self._attached.append(logger)
            self._status = "direct"
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
            "error": self._error,
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
        for capture in captures
    )
    hot_zero_compile = all(
        capture.get("compile_event_count") == 0 for capture in (hot1, hot2)
    )
    hot_no_miss = all(
        capture.get("persistent_cache_misses") == 0 for capture in (hot1, hot2)
    )
    passed = direct and hot_zero_compile and hot_no_miss
    return {
        "hot_cache_reuse": passed,
        "hot_prefill_compile_seconds": 0.0 if passed else None,
        "hot_decode_compile_seconds": 0.0 if passed else None,
        "compile_evidence": "PASS" if passed else "FAIL",
        "reason": (
            "direct zero-compile evidence for HOT-1 and HOT-2"
            if passed
            else "direct compile/cache evidence was incomplete or showed HOT compilation"
        ),
    }
