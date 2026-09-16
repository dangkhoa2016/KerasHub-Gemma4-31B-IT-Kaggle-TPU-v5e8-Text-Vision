#!/usr/bin/env python3
"""Single-model native Gemma4 image-conditioned G5 authority process."""

from __future__ import annotations

import argparse
import gc
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
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                values[parts[0]] = int(parts[1])
        return values
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
    def __init__(self, seconds, marker, result_path, result):
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

    def _run(self):
        if self.cancelled.wait(self.seconds):
            return
        self.result[self.marker] = True
        self.result["FINAL_RESULT"] = self.marker
        _write_result(self.result_path, self.result)
        os._exit(124)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.cancelled.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    evidence_dir = args.evidence_dir
    result_path = evidence_dir / "11-g5-result.json"
    result: dict[str, object] = {
        "AUTHORITY_NOT_STARTED": False,
        "G5_PATH": "NATIVE_VISION",
        "TPU_INITIALIZATION_ATTEMPTS": 1,
        "TPU_DEVICE_COUNT": None,
        "MODEL_LOAD_COUNT": 1,
        "MODEL_INSTANCE_COUNT": 1,
        "MODEL_LOAD_STARTED": False,
        "MODEL_LOAD_RETURNED": False,
        "CANDIDATE_A_SHARDING_VERIFIED": False,
        "IMAGE_INPUT_LOADED": False,
        "IMAGE_PREPROCESS_RETURNED": False,
        "VISION_CONDITIONING_PRESENT": False,
        "G5_GENERATION_STARTED": False,
        "G5_GENERATION_RETURNED": False,
        "G5_GENERATION_CALL_COUNT": 0,
        "G5_IMAGE_GENERATION": "NOT_EVALUATED",
        "G5_STATUS": "OPEN",
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
        from gemma4_server.tpu.g5_vision import (
            G5_PROMPT,
            G5_SEQUENCE_LENGTH,
            inspect_vision_preprocess,
            load_fixture_rgb,
            prepare_image_inputs,
            validate_non_empty_generation,
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
            generation_length_buckets=(512, 768, 1024, 1536, 2048),
            max_generation_length=2048,
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
        result["MODEL_LOAD_METADATA"] = engine.metadata
        _write_result(result_path, result)
        print("MODEL_LOAD_RETURNED=true", flush=True)

        candidate = verify_candidate_a(engine.model)
        result.update(candidate)
        result["CANDIDATE_A_SHARDING_VERIFIED"] = True
        result["CGROUP_AFTER_LOAD"] = cgroup_snapshot()
        _write_result(result_path, result)

        fixture_path = Path(os.environ["G5_IMAGE_FIXTURE_PATH"])
        image = load_fixture_rgb(fixture_path)
        result["IMAGE_INPUT_LOADED"] = True
        result["IMAGE_FIXTURE_PATH"] = str(fixture_path)
        result["IMAGE_FIXTURE_SHA256"] = os.environ.get(
            "G5_IMAGE_FIXTURE_SHA256", "unknown"
        )
        processed, rendered_prompt = prepare_image_inputs(
            engine.preprocessor,
            image,
            G5_PROMPT,
            sequence_length=G5_SEQUENCE_LENGTH,
        )
        vision_summary = inspect_vision_preprocess(processed)
        result.update(
            {
                "IMAGE_PREPROCESS_RETURNED": vision_summary[
                    "image_preprocess_returned"
                ],
                "VISION_CONDITIONING_PRESENT": vision_summary[
                    "vision_conditioning_present"
                ],
                "VISION_PREPROCESS_SUMMARY": vision_summary,
                "PROMPT_TEXT": G5_PROMPT,
                "RENDERED_PROMPT": rendered_prompt,
            }
        )
        del processed
        gc.collect()
        result["CGROUP_AFTER_PREPROCESS"] = cgroup_snapshot()
        _write_result(result_path, result)

        if not result["IMAGE_PREPROCESS_RETURNED"]:
            raise RuntimeError("G5 image preprocessing did not return its fields")
        if not result["VISION_CONDITIONING_PRESENT"]:
            raise RuntimeError("G5 image preprocessing returned no vision conditioning")

        result["G5_GENERATION_STARTED"] = True
        result["G5_GENERATION_CALL_COUNT"] = 1
        result["CGROUP_BEFORE_GENERATION"] = cgroup_snapshot()
        _write_result(result_path, result)
        print("G5_GENERATION_STARTED=true", flush=True)
        started = time.perf_counter()
        output, metrics = engine.generate_image(
            image,
            G5_PROMPT,
            "",
            1,
        )
        elapsed = time.perf_counter() - started
        result["G5_GENERATION_RETURNED"] = True
        result["G5_GENERATION_RESULT"] = validate_non_empty_generation(output)
        result["G5_GENERATION_SECONDS"] = round(elapsed, 6)
        result["GENERATION_METRICS"] = metrics
        result["CGROUP_AFTER_GENERATION"] = cgroup_snapshot()
        result["G5_IMAGE_GENERATION"] = "PASS"
        result["G5_STATUS"] = "CLOSED"
        result["G6_ENTRY_ELIGIBLE"] = True
        result["FINAL_RESULT"] = "G5_IMAGE_CONDITIONED_GENERATION_PASS"
        print("G5_GENERATION_RETURNED=true", flush=True)
        return 0
    except CandidateAVerificationError as exc:
        result["ERROR_TYPE"] = type(exc).__name__
        result["ERROR"] = str(exc)
        result["FINAL_RESULT"] = "G5_CANDIDATE_A_DRIFT"
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
            result["FINAL_RESULT"] = "G5_CANDIDATE_A_VERIFICATION_FAILED"
        elif not result["G5_GENERATION_STARTED"]:
            result["FINAL_RESULT"] = "G5_VISION_PREPROCESS_FAILED"
        else:
            result["FINAL_RESULT"] = "G5_IMAGE_GENERATION_FAILED"
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
