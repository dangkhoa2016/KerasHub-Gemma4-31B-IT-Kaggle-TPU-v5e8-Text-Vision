#!/usr/bin/env python3
"""CPU-only frozen-baseline and native-vs-split evidence helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
from typing import Any


FROZEN_G3_ARCHIVE_SHA256 = (
    "475c940baf60302fd0f03e5f4b1a156696598d51162310ff168283551d6bb38e"
)


def _frozen_result(archive_path: str | Path) -> dict[str, Any]:
    archive = Path(archive_path)
    with tarfile.open(archive, mode="r:gz") as stream:
        member = next(
            (
                item
                for item in stream.getmembers()
                if item.name.endswith("/07-authority-result.json")
            ),
            None,
        )
        if member is None:
            raise ValueError("frozen G3 archive has no authority result")
        extracted = stream.extractfile(member)
        if extracted is None:
            raise ValueError("frozen G3 authority result is unreadable")
        value = json.load(extracted)
    if not isinstance(value, dict):
        raise ValueError("frozen G3 authority result is not a JSON object")
    return value


def native_baseline_from_archive(archive_path: str | Path) -> dict[str, Any]:
    """Extract only directly supported native metrics from frozen G3."""
    frozen = _frozen_result(archive_path)
    metadata = frozen.get("GENERATION_METADATA") or {}
    if frozen.get("FINAL_RESULT") != "G3_PASS":
        raise ValueError("frozen G3 result is not PASS")
    baseline = {
        "source": "frozen-g3-authority",
        "authority_archive": str(archive_path),
        "authority_archive_sha256": FROZEN_G3_ARCHIVE_SHA256,
        "prompt_text": frozen.get("PROMPT_TEXT"),
        "prompt_tokens": frozen.get("PROMPT_TOKEN_COUNT"),
        "requested_new_tokens": frozen.get("MAX_NEW_TOKENS"),
        "authority_max_length": frozen.get("AUTHORITY_MAX_LENGTH"),
        "generation_seconds": metadata.get("generation_seconds"),
        "generation_call_count": frozen.get("GENERATION_CALL_COUNT"),
        "tpu_device_count": frozen.get("TPU_DEVICE_COUNT"),
        "candidate_a_sharding_verified": frozen.get(
            "CANDIDATE_A_SHARDING_VERIFIED"
        ),
        "run_eagerly": True,
    }
    required = (
        "prompt_text",
        "prompt_tokens",
        "requested_new_tokens",
        "authority_max_length",
        "generation_seconds",
        "generation_call_count",
        "tpu_device_count",
        "candidate_a_sharding_verified",
    )
    if any(baseline[key] is None for key in required):
        raise ValueError("frozen G3 evidence is missing a required metric")
    return baseline


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_comparison(
    path: str | Path,
    native: dict[str, Any],
    split: dict[str, Any],
    *,
    split_host_oom_delta: int | None = None,
) -> dict[str, Any]:
    """Write comparison metrics without manufacturing missing measurements."""
    comparison = {
        "native_source": native["source"],
        "native_generation_seconds": native["generation_seconds"],
        "split_result": split.get("G4_SPLIT_MINIMAL_RUN", "FAIL"),
        "split_generation_seconds": split.get(
            "G4_SPLIT_GENERATION_SECONDS"
        ),
        "split_host_oom_delta": split_host_oom_delta,
        "split_tpu_device_count": split.get("TPU_DEVICE_COUNT"),
        "split_model_load_count": split.get("MODEL_LOAD_COUNT"),
        "memory_comparison_status": "PARTIAL",
    }
    write_json(path, comparison)
    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, native_baseline_from_archive(args.archive))
