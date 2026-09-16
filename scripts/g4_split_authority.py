#!/usr/bin/env python3
"""The single JAX-importing G4 native split authority process."""

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
        json.dump(result, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_events(path: Path) -> dict[str, int]:
    try:
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = int(parts[1])
        return result
    except (OSError, ValueError):
        return {}


def _cgroup_paths() -> tuple[Path, Path, Path]:
    directory = Path("/sys/fs/cgroup")
    try:
        for line in Path("/proc/self/cgroup").read_text(
            encoding="utf-8"
        ).splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                directory = Path("/sys/fs/cgroup") / path.lstrip("/")
                break
    except (OSError, ValueError):
        pass
    return (
        directory / "memory.events",
        directory / "memory.current",
        directory / "memory.max",
    )


def cgroup_snapshot() -> dict[str, object]:
    events_path, current_path, maximum_path = _cgroup_paths()
    result: dict[str, object] = {
        "memory_events_path": str(events_path),
        "memory_current_path": str(current_path),
        "memory_max_path": str(maximum_path),
        "memory_events": _read_events(events_path),
    }
    for key, path in (
        ("memory_current", current_path),
        ("memory_max", maximum_path),
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            result[key] = raw if raw == "max" else int(raw)
        except (OSError, ValueError):
            result[key] = None
    return result


class HardDeadline:
    def __init__(
        self,
        seconds: int,
        marker: str,
        result_path: Path,
        result: dict[str, object],
    ):
        self.seconds = int(seconds)
        self.marker = marker
        self.result_path = result_path
        self.result = result
        self.cancelled = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"{marker.lower()}-watchdog",
            daemon=True,
        )

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
        if self.thread.is_alive():
            self.thread.join(timeout=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    evidence_dir = args.evidence_dir
    result_path = evidence_dir / "07-g4-result.json"
    result: dict[str, object] = {
        "AUTHORITY_NOT_STARTED": False,
        "TPU_INITIALIZATION_ATTEMPTS": 1,
        "TPU_DEVICE_COUNT": None,
        "MODEL_LOAD_COUNT": 1,
        "MODEL_INSTANCE_COUNT": 1,
        "MODEL_LOAD_STARTED": False,
        "MODEL_LOAD_RETURNED": False,
        "CANDIDATE_A_SHARDING_VERIFIED": False,
        "G4_SPLIT_PATH_STARTED": False,
        "G4_SPLIT_PATH_RETURNED": False,
        "G4_SPLIT_GENERATION_CALL_COUNT": 0,
        "G4_SPLIT_MINIMAL_RUN": "NOT_EVALUATED",
        "G4_STATUS": "OPEN",
        "FINAL_RESULT": "AUTHORITY_NOT_FINISHED",
    }
    _write_result(result_path, result)
    from gemma4_server.tpu.authority_contract import CandidateAVerificationError

    init_started = time.monotonic()
    init_watchdog = HardDeadline(
        TPU_INIT_HARD_MAX_SECONDS,
        "TPU_INITIALIZATION_TIMEOUT",
        result_path,
        result,
    )
    load_watchdog = None
    try:
        init_watchdog.start()
        import jax
        import keras

        devices = list(jax.devices("tpu"))
        result["TPU_DEVICE_COUNT"] = len(devices)
        if len(devices) != 8:
            result["FINAL_RESULT"] = "TPU_DEVICE_COUNT_MISMATCH"
            _write_result(result_path, result)
            return 1
        result["TPU_INITIALIZATION_SECONDS"] = round(
            time.monotonic() - init_started, 6
        )
        result["TPU_INITIALIZATION_TARGET_SECONDS"] = TPU_INIT_TARGET_SECONDS
        result["TPU_INITIALIZATION_HARD_MAX_SECONDS"] = TPU_INIT_HARD_MAX_SECONDS
        result["CGROUP_BEFORE_AUTHORITY"] = cgroup_snapshot()
        init_watchdog.cancel()
        print("TPU_DEVICE_COUNT=8", flush=True)

        from gemma4_server.tpu.authority_contract import verify_candidate_a
        from gemma4_server.tpu.distribution import build_distribution
        from gemma4_server.tpu.engine import Gemma4TPUEngine
        from gemma4_server.tpu.g4_split import (
            G4_MAX_LENGTH,
            G4_PROMPT,
            run_split_generation,
        )

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
            generation_length_buckets=(G4_MAX_LENGTH,),
            max_generation_length=G4_MAX_LENGTH,
            min_sharded_parameter_percent=80.0,
        )
        load_watchdog = HardDeadline(
            STRICT_LOAD_HARD_MAX_SECONDS,
            "STRICT_LOAD_TIMEOUT",
            result_path,
            result,
        )
        result["MODEL_LOAD_STARTED"] = True
        result["MODEL_LOAD_COUNT"] = 1
        _write_result(result_path, result)
        print("MODEL_LOAD_STARTED=true", flush=True)
        load_watchdog.start()
        engine.load()
        load_watchdog.cancel()
        result["MODEL_LOAD_RETURNED"] = True
        result["MODEL_LOAD_METADATA"] = engine.metadata
        _write_result(result_path, result)
        print("MODEL_LOAD_RETURNED=true", flush=True)

        candidate = verify_candidate_a(engine.model)
        result.update(candidate)
        result["CANDIDATE_A_SHARDING_VERIFIED"] = True
        result["CGROUP_AFTER_LOAD"] = cgroup_snapshot()
        _write_result(result_path, result)

        result["G4_SPLIT_PATH_STARTED"] = True
        result["G4_SPLIT_GENERATION_CALL_COUNT"] = 1
        _write_result(result_path, result)
        print("G4_SPLIT_PATH_STARTED=true", flush=True)
        split_result = run_split_generation(engine.model, G4_PROMPT)
        result["G4_SPLIT_PATH_RETURNED"] = True
        result["G4_SPLIT_MINIMAL_RUN"] = "PASS"
        result["G4_SPLIT_GENERATION_RESULT"] = split_result["result"]
        result["G4_SPLIT_GENERATION_SECONDS"] = split_result[
            "split_generation_seconds"
        ]
        result["PROMPT_TEXT"] = split_result["prompt_text"]
        result["PROMPT_TOKEN_COUNT"] = split_result["prompt_tokens"]
        result["REQUESTED_NEW_TOKENS"] = split_result["requested_new_tokens"]
        result["AUTHORITY_MAX_LENGTH"] = split_result["max_length"]
        result["CGROUP_AFTER_GENERATION"] = cgroup_snapshot()
        result["FINAL_RESULT"] = "G4_SPLIT_MINIMAL_RUN_PASS"
        result["G4_STATUS"] = "CLOSED"
        print("G4_SPLIT_PATH_RETURNED=true", flush=True)
        return 0
    except CandidateAVerificationError as exc:
        result["ERROR_TYPE"] = type(exc).__name__
        result["ERROR"] = str(exc)
        result["FINAL_RESULT"] = "G4_CANDIDATE_A_DRIFT"
        _write_result(result_path, result)
        return 1
    except BaseException as exc:
        result["ERROR_TYPE"] = type(exc).__name__
        result["ERROR"] = str(exc)
        if not result["MODEL_LOAD_RETURNED"]:
            result["FINAL_RESULT"] = (
                "STRICT_LOAD_FAILED"
                if result["MODEL_LOAD_STARTED"]
                else "AUTHORITY_FAILED"
            )
        elif not result["CANDIDATE_A_SHARDING_VERIFIED"]:
            result["FINAL_RESULT"] = "G4_CANDIDATE_A_VERIFICATION_FAILED"
        else:
            result["FINAL_RESULT"] = "G4_SPLIT_PATH_FAILED"
        _write_result(result_path, result)
        print(f"FINAL_RESULT={result['FINAL_RESULT']}", flush=True)
        return 1
    finally:
        init_watchdog.cancel()
        if load_watchdog is not None:
            load_watchdog.cancel()
        result["CGROUP_AFTER_AUTHORITY"] = cgroup_snapshot()
        _write_result(result_path, result)


if __name__ == "__main__":
    raise SystemExit(main())
