#!/usr/bin/env python3
"""The single JAX-importing G3 TPU authority process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time


TPU_INIT_TARGET_SECONDS = 120
TPU_INIT_HARD_MAX_SECONDS = 180
STRICT_LOAD_HARD_MAX_SECONDS = 2304


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_events(path: Path) -> dict[str, int]:
    try:
        return {
            key: int(value)
            for key, value in (
                line.split(None, 1)
                for line in path.read_text(encoding="utf-8").splitlines()
                if len(line.split(None, 1)) == 2
            )
        }
    except (OSError, ValueError):
        return {}


def _cgroup_paths() -> tuple[Path, Path, Path]:
    directory = Path("/sys/fs/cgroup")
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                directory = Path("/sys/fs/cgroup") / path.lstrip("/")
                break
    except (OSError, ValueError):
        pass
    return directory / "memory.events", directory / "memory.current", directory / "memory.max"


def _cgroup_snapshot() -> dict[str, object]:
    events_path, current_path, maximum_path = _cgroup_paths()
    events = _read_events(events_path)
    values: dict[str, object] = {
        "memory_events_path": str(events_path),
        "memory_current_path": str(current_path),
        "memory_max_path": str(maximum_path),
        "memory_events": events,
    }
    for key, path in (("memory_current", current_path), ("memory_max", maximum_path)):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            values[key] = raw if raw == "max" else int(raw)
        except (OSError, ValueError):
            values[key] = None
    return values


def _memory_cleanup_fields(cleanup: dict[str, object]) -> dict[str, object]:
    return {
        "HOST_MEMORY_BEFORE_CLEANUP": cleanup.get("post_load_rss_before_cleanup_kib"),
        "HOST_MEMORY_AFTER_GC": cleanup.get("post_load_rss_after_gc_kib"),
        "HOST_MEMORY_AFTER_MALLOC_TRIM": cleanup.get("post_load_rss_after_malloc_trim_kib"),
        "MALLOC_TRIM_AVAILABLE": cleanup.get("post_load_malloc_trim_available"),
        "MALLOC_TRIM_RESULT": cleanup.get("post_load_malloc_trim_result"),
    }


class HardDeadline:
    def __init__(self, seconds: int, marker: str, result_path: Path, result: dict[str, object]):
        self.seconds = int(seconds)
        self.marker = marker
        self.result_path = result_path
        self.result = result
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"{marker.lower()}-watchdog", daemon=True)

    def _run(self) -> None:
        if self.cancelled.wait(self.seconds):
            return
        self.result[self.marker] = True
        self.result["FINAL_RESULT"] = self.marker
        _write_result(self.result_path, self.result)
        os._exit(124)

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.cancelled.set()
        self.thread.join(timeout=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    evidence_dir = Path(args.evidence_dir)
    result_path = evidence_dir / "07-authority-result.json"
    prompt = "Hello"
    prompt_token_count = 10
    max_new_tokens = 1
    authority_max_length = 11
    generation_call_count = 0
    init_started = time.monotonic()
    result: dict[str, object] = {
        "AUTHORITY_NOT_STARTED": False,
        "TPU_INITIALIZATION_ATTEMPTS": 1,
        "TPU_DEVICE_COUNT": None,
        "MODEL_LOAD_STARTED": False,
        "MODEL_LOAD_RETURNED": False,
        "CANDIDATE_A_SHARDING_VERIFIED": False,
        "GENERATION_CALL_COUNT": 0,
        "GENERATION_STARTED": False,
        "GENERATION_RETURNED": False,
        "GENERATION_RESULT": None,
        "G3_STATUS": "OPEN",
        "G3_TEXT_GENERATION": "NOT_EVALUATED",
        "G4_ENTRY_ELIGIBLE": False,
        "G4_STARTED": False,
        "FINAL_RESULT": "AUTHORITY_NOT_FINISHED",
    }
    _write_result(result_path, result)
    init_watchdog = HardDeadline(
        TPU_INIT_HARD_MAX_SECONDS,
        "TPU_INIT_TIMEOUT",
        result_path,
        result,
    )
    init_watchdog.start()
    load_watchdog = None
    try:
        import jax
        import keras
        import keras_hub

        devices = list(jax.devices("tpu"))
        result["TPU_DEVICE_COUNT"] = len(devices)
        if len(devices) != 8:
            init_watchdog.cancel()
            result["FINAL_RESULT"] = "TPU_DEVICE_COUNT_MISMATCH"
            _write_result(result_path, result)
            return 1
        init_watchdog.cancel()
        result["TPU_INITIALIZATION_SECONDS"] = round(
            time.monotonic() - init_started, 6
        )
        result["TPU_INITIALIZATION_TARGET_SECONDS"] = TPU_INIT_TARGET_SECONDS
        result["TPU_INITIALIZATION_HARD_MAX_SECONDS"] = TPU_INIT_HARD_MAX_SECONDS
        result["CGROUP_BEFORE_AUTHORITY"] = _cgroup_snapshot()
        print("TPU_DEVICE_COUNT=8", flush=True)

        from gemma4_server.tpu.authority_contract import verify_candidate_a
        from gemma4_server.tpu.distribution import build_distribution
        from gemma4_server.tpu.engine import Gemma4TPUEngine, post_load_host_cleanup

        _, _, distribution = build_distribution(
            keras,
            jax,
            shape=(1, 8),
            axis_names=("batch", "model"),
        )
        engine = Gemma4TPUEngine(
            os.environ["MODEL_PATH"],
            "bfloat16",
            distribution,
            vision_enabled=True,
            generation_length_buckets=(11,),
            max_generation_length=authority_max_length,
            min_sharded_parameter_percent=80.0,
        )

        load_watchdog = HardDeadline(
            STRICT_LOAD_HARD_MAX_SECONDS,
            "STRICT_LOAD_TIMEOUT",
            result_path,
            result,
        )
        result["MODEL_LOAD_STARTED"] = True
        _write_result(result_path, result)
        print("MODEL_LOAD_STARTED=true", flush=True)
        load_watchdog.start()
        engine.load()
        load_watchdog.cancel()
        result["MODEL_LOAD_RETURNED"] = True
        _write_result(result_path, result)
        print("MODEL_LOAD_RETURNED=true", flush=True)

        candidate = verify_candidate_a(engine.model)
        result.update(candidate)
        result["CANDIDATE_A_SHARDING_VERIFIED"] = True
        _write_result(result_path, result)

        cleanup = post_load_host_cleanup()
        result.update(_memory_cleanup_fields(cleanup))
        result["CGROUP_AFTER_CLEANUP"] = _cgroup_snapshot()
        _write_result(result_path, result)

        if generation_call_count >= 1:
            raise RuntimeError("generation call count exceeded one")
        generation_call_count = 1
        result["GENERATION_CALL_COUNT"] = generation_call_count
        result["GENERATION_STARTED"] = True
        result["PROMPT_TEXT"] = prompt
        result["PROMPT_TOKEN_COUNT"] = prompt_token_count
        result["MAX_NEW_TOKENS"] = max_new_tokens
        result["AUTHORITY_MAX_LENGTH"] = authority_max_length
        _write_result(result_path, result)
        print("GENERATION_STARTED=true", flush=True)
        output, generation_metadata = engine.generate_text_authority(
            prompt,
            "",
            max_new_tokens,
        )
        result["GENERATION_RETURNED"] = True
        result["GENERATION_RESULT"] = output
        result["GENERATION_METADATA"] = generation_metadata
        if not output:
            raise RuntimeError("authority generation returned empty output")
        result["G3_STATUS"] = "CLOSED"
        result["G3_TEXT_GENERATION"] = "PASS"
        result["G4_ENTRY_ELIGIBLE"] = True
        result["FINAL_RESULT"] = "G3_PASS"
        print("GENERATION_RETURNED=true", flush=True)
        return 0
    except BaseException as exc:
        result["ERROR_TYPE"] = type(exc).__name__
        result["ERROR"] = str(exc)
        if not result["MODEL_LOAD_RETURNED"]:
            result["FINAL_RESULT"] = "STRICT_LOAD_FAILED" if result["MODEL_LOAD_STARTED"] else "AUTHORITY_FAILED"
        elif not result["CANDIDATE_A_SHARDING_VERIFIED"]:
            result["FINAL_RESULT"] = "CANDIDATE_A_VERIFICATION_FAILED"
        else:
            result["FINAL_RESULT"] = "G3_GENERATION_FAILED"
        _write_result(result_path, result)
        print(f"FINAL_RESULT={result['FINAL_RESULT']}", flush=True)
        return 1
    finally:
        init_watchdog.cancel()
        if load_watchdog is not None:
            load_watchdog.cancel()
        result["CGROUP_AFTER_AUTHORITY"] = _cgroup_snapshot()
        _write_result(result_path, result)


if __name__ == "__main__":
    raise SystemExit(main())
