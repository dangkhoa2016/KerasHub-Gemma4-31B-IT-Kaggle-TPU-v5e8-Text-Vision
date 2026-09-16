#!/usr/bin/env python3
"""Adjudicate authority OOM stage from the structured authority result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GENERATION_OOM = "G3_EXACT_LENGTH_GENERATION_HOST_OOM"
PRE_GENERATION_OOM = "AUTHORITY_PRE_GENERATION_HOST_OOM"
UNKNOWN_STAGE_OOM = "AUTHORITY_HOST_OOM_STAGE_UNKNOWN"


def _read_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _generation_state(result: dict[str, Any]) -> str:
    marker = result.get("GENERATION_STARTED")
    if type(marker) is bool:
        return "true" if marker else "false"
    return "unknown"


def adjudicate(
    authority_exit: int,
    oom_delta: int,
    result: dict[str, Any],
) -> dict[str, str]:
    state = _generation_state(result)
    if authority_exit == 137 and oom_delta > 0:
        if state == "true":
            return {
                "FINAL_RESULT": GENERATION_OOM,
                "GENERATION_STARTED": state,
                "G3_TEXT_GENERATION": "FAIL",
                "G3_STATUS": "OPEN",
                "G4_ENTRY_ELIGIBLE": "false",
                "G4_STARTED": "false",
            }
        if state == "false":
            return {
                "FINAL_RESULT": PRE_GENERATION_OOM,
                "GENERATION_STARTED": state,
                "G3_TEXT_GENERATION": "NOT_EVALUATED",
                "G3_STATUS": "OPEN",
                "G4_ENTRY_ELIGIBLE": "false",
                "G4_STARTED": "false",
            }
        return {
            "FINAL_RESULT": UNKNOWN_STAGE_OOM,
            "GENERATION_STARTED": "unknown",
            "G3_TEXT_GENERATION": "NOT_EVALUATED",
            "G3_STATUS": "OPEN",
            "G4_ENTRY_ELIGIBLE": "false",
            "G4_STARTED": "false",
        }

    final_result = result.get("FINAL_RESULT")
    if not isinstance(final_result, str) or not final_result:
        final_result = "AUTHORITY_PROCESS_FAILED"
    return {
        "FINAL_RESULT": final_result,
        "GENERATION_STARTED": state,
        "G3_TEXT_GENERATION": str(result.get("G3_TEXT_GENERATION", "NOT_EVALUATED")),
        "G3_STATUS": str(result.get("G3_STATUS", "OPEN")),
        "G4_ENTRY_ELIGIBLE": str(result.get("G4_ENTRY_ELIGIBLE", False)).lower(),
        "G4_STARTED": str(result.get("G4_STARTED", False)).lower(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-exit", type=int, required=True)
    parser.add_argument("--oom-delta", type=int, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = adjudicate(args.authority_exit, args.oom_delta, _read_result(args.result_json))
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
