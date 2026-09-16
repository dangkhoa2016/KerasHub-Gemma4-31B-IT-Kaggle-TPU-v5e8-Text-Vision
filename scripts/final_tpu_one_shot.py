#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import argparse
import base64
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


OOM_KEYS = ("oom", "oom_kill", "oom_group_kill")
FORBIDDEN_DIRS = {".git", "cache", "caches", "state", "weights"}
FORBIDDEN_SUFFIXES = {".log", ".pid"}
FORBIDDEN_MARKERS = ("secret", "credential", "api_key", "token")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def runtime_identity_from_paths(
    *,
    hostname_path: Path,
    boot_id_path: Path,
    pid1_path: Path,
) -> dict[str, str | None]:
    return {
        "hostname": _read_text(hostname_path),
        "boot_id": _read_text(boot_id_path),
        "pid1": _read_text(pid1_path),
    }


def read_runtime_identity() -> dict[str, str | None]:
    return runtime_identity_from_paths(
        hostname_path=Path("/etc/hostname"),
        boot_id_path=Path("/proc/sys/kernel/random/boot_id"),
        pid1_path=Path("/proc/1/stat"),
    ) | {
        "pid": str(os.getpid()),
        "jupyter_parent_pid": os.environ.get("JPY_PARENT_PID"),
    }


def session_fresh_from_checkpoint(path: Path) -> tuple[bool, dict[str, Any]]:
    """Compare current runtime identity with the pre-restart G9 checkpoint."""
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    baseline = checkpoint.get("runtime_identity_before", checkpoint)
    current = read_runtime_identity()
    comparable = (
        "boot_id",
        "hostname",
        "pid1",
        "jupyter_parent_pid",
    )
    changed = {
        key: {
            "before": baseline.get(key),
            "after": current.get(key),
        }
        for key in comparable
        if baseline.get(key) is not None
        and current.get(key) is not None
        and baseline.get(key) != current.get(key)
    }
    return bool(changed), {
        "checkpoint": str(path),
        "runtime_identity_before": baseline,
        "runtime_identity_after": current,
        "changed_fields": changed,
    }


def _read_int(path: Path) -> int | str | None:
    value = _read_text(path)
    if value is None:
        return None
    if value == "max":
        return value
    try:
        return int(value)
    except ValueError:
        return None


def _read_events(path: Path) -> dict[str, int | None]:
    values = {key: None for key in OOM_KEYS}
    text = _read_text(path)
    if text is None:
        return values
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] not in values:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def read_cgroup_snapshot(
    root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    return {
        "memory_current": _read_int(root / "memory.current"),
        "memory_max": _read_int(root / "memory.max"),
        "memory_events": _read_events(root / "memory.events"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_source_identity(repo: Path, expected_sha: str) -> dict[str, Any]:
    if not expected_sha or len(expected_sha) != 40:
        raise ValueError("expected_sha must be a 40-character SHA")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != expected_sha:
        raise RuntimeError(f"source is not exact: {head} != {expected_sha}")
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError("source tree is not clean")
    return {
        "head": head,
        "expected_sha": expected_sha,
        "git_sha_exact": True,
        "worktree_clean": True,
    }


def request_json(
    base_url: str,
    path: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    body = None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload_bytes = exc.read()
        try:
            response_payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            response_payload = {"error": payload_bytes.decode("utf-8", "replace")}
        return exc.code, response_payload


def poll_job(
    base_url: str,
    job_id: str,
    headers: dict[str, str],
    timeout: float,
    interval: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_code, payload = request_json(
            base_url,
            f"/result/{job_id}",
            headers=headers,
            timeout=min(30.0, max(0.1, deadline - time.monotonic())),
        )
        status = payload.get("status")
        if status in {"completed", "failed"}:
            payload["http_status"] = status_code
            return payload
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"timed out polling job {job_id}")


def _is_forbidden(path: Path) -> bool:
    if any(part in FORBIDDEN_DIRS for part in path.parts):
        return True
    name = path.name.lower()
    if name == "sha256sums" or name.startswith("."):
        return True
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    return any(marker in name for marker in FORBIDDEN_MARKERS)


def package_evidence(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not _is_forbidden(path.relative_to(directory))
    )
    sums_path = directory / "SHA256SUMS"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in files
    ]
    sums_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return sums_path


class RestClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        timeout: float = 30.0,
    ):
        self.base_url = base_url
        self.headers = headers
        self.timeout = timeout

    def get(self, path: str) -> tuple[int, dict[str, Any]]:
        return request_json(
            self.base_url,
            path,
            headers=self.headers,
            timeout=self.timeout,
        )

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return request_json(
            self.base_url,
            path,
            method="POST",
            payload=payload,
            headers=self.headers,
            timeout=self.timeout,
        )

    def poll(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        return poll_job(
            self.base_url,
            job_id,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _nonempty_output(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("output"), str) and bool(payload["output"].strip())


def _submit_and_poll(
    client,
    path: str,
    payload: dict[str, Any],
    post_counter: list[int] | None = None,
) -> dict[str, Any]:
    if post_counter is not None:
        post_counter[0] += 1
    status_code, accepted = client.post(path, payload)
    if status_code != 202 or not accepted.get("job_id"):
        raise RuntimeError(f"async request was not accepted: {path} {status_code}")
    result = client.poll(accepted["job_id"])
    if result.get("status") != "completed" or not _nonempty_output(result):
        raise RuntimeError(f"job did not complete with nonempty output: {result}")
    return {
        "request": payload,
        "accepted_response": accepted,
        "result": result,
    }


def run_text_acceptance(
    client,
    evidence_dir: Path,
    label: str,
    post_counter: list[int] | None = None,
) -> dict[str, Any]:
    payload = {
        "prompt": "hello",
        "system": "",
        "max_new_tokens": 1,
    }
    try:
        record = _submit_and_poll(
            client,
            "/generate/async",
            payload,
            post_counter,
        )
        result = {
            "label": label,
            "accepted": True,
            "result_nonempty": True,
            **record,
        }
    except Exception as exc:
        result = {
            "label": label,
            "accepted": False,
            "result_nonempty": False,
            "error": repr(exc),
        }
    _write_json(evidence_dir / f"{label}.json", result)
    return result


def run_vision_acceptance(client, evidence_dir: Path, label: str) -> dict[str, Any]:
    from gemma4_server.tpu.g5_vision import create_synthetic_fixture

    try:
        with tempfile.TemporaryDirectory(prefix="g9-vision-") as tmp:
            fixture = Path(tmp) / "g5-fixture.png"
            create_synthetic_fixture(fixture)
            image_base64 = base64.b64encode(fixture.read_bytes()).decode("ascii")
        payload = {
            "prompt": "Describe the image.",
            "system": "",
            "max_new_tokens": 1,
            "image_base64": image_base64,
        }
        record = _submit_and_poll(client, "/generate/image/async", payload)
        metrics = record["result"].get("metrics") or {}
        conditioning = metrics.get("vision_conditioning_present") is True
        result = {
            "label": label,
            "accepted": conditioning,
            "result_nonempty": True,
            "vision_conditioning_present": conditioning,
            **record,
        }
        if not conditioning:
            result["error"] = "vision conditioning was not directly evidenced"
    except Exception as exc:
        result = {
            "label": label,
            "accepted": False,
            "result_nonempty": False,
            "vision_conditioning_present": False,
            "error": repr(exc),
        }
    _write_json(evidence_dir / f"{label}.json", result)
    return result


def adjudicate_g9(rows: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(rows)
    required = {
        "prime": bool(rows.get("prime", rows.get("G9_PRIME") == "PASS")),
        "hot_1": bool(rows.get("hot_1", rows.get("G9_HOT_1") == "PASS")),
        "hot_2": bool(rows.get("hot_2", rows.get("G9_HOT_2") == "PASS")),
        "hot_cache_reuse": bool(
            rows.get("hot_cache_reuse", rows.get("G9_HOT_CACHE_REUSE", False))
        ),
        "compile_evidence": rows.get(
            "compile_evidence", rows.get("G9_COMPILE_EVIDENCE", "FAIL")
        ),
        "text_semantic_acceptance": bool(
            rows.get("text_semantic_acceptance", False)
        ),
        "vision_semantic_acceptance": bool(
            rows.get("vision_semantic_acceptance", False)
        ),
        "rest_acceptance": bool(rows.get("rest_acceptance", False)),
        "oom_delta": rows.get("oom_delta"),
        "pre_prime_authority_gate": rows.get(
            "PRE_PRIME_AUTHORITY_GATE", "FAIL"
        )
        == "PASS",
        "generation_post_count": rows.get("GENERATION_POST_COUNT", 0),
        "hot_prefill_compile_seconds": rows.get(
            "hot_prefill_compile_seconds",
            rows.get("HOT_PREFILL_COMPILE_SECONDS"),
        ),
        "hot_decode_compile_seconds": rows.get(
            "hot_decode_compile_seconds",
            rows.get("HOT_DECODE_COMPILE_SECONDS"),
        ),
    }
    passed = all(
        (
            required["prime"],
            required["hot_1"],
            required["hot_2"],
            required["hot_cache_reuse"],
            required["compile_evidence"] == "PASS",
            required["text_semantic_acceptance"],
            required["vision_semantic_acceptance"],
            required["rest_acceptance"],
            required["oom_delta"] == 0,
            required["pre_prime_authority_gate"],
            required["generation_post_count"] > 0,
            required["hot_prefill_compile_seconds"] == 0.0,
            required["hot_decode_compile_seconds"] == 0.0,
        )
    )
    normalized.update(required)
    normalized.update(
        {
            "G9_PRIME": "PASS" if required["prime"] else "FAIL",
            "G9_HOT_1": "PASS" if required["hot_1"] else "FAIL",
            "G9_HOT_2": "PASS" if required["hot_2"] else "FAIL",
            "G9_HOT_CACHE_REUSE": required["hot_cache_reuse"],
            "G9_COMPILE_EVIDENCE": required["compile_evidence"],
            "PRE_PRIME_AUTHORITY_GATE": (
                "PASS" if required["pre_prime_authority_gate"] else "FAIL"
            ),
            "GENERATION_POST_COUNT": required["generation_post_count"],
            "HOT_CACHE_REUSE": required["hot_cache_reuse"],
            "HOT_PREFILL_COMPILE_SECONDS": required[
                "hot_prefill_compile_seconds"
            ],
            "HOT_DECODE_COMPILE_SECONDS": required[
                "hot_decode_compile_seconds"
            ],
            "G9_STATUS": "CLOSED/PASS" if passed else "OPEN/FAIL",
        }
    )
    return normalized


def _oom_delta(before: dict[str, Any], after: dict[str, Any]) -> int | str:
    before_events = before.get("memory_events", {})
    after_events = after.get("memory_events", {})
    deltas = []
    for key in OOM_KEYS:
        if before_events.get(key) is None or after_events.get(key) is None:
            return "NOT_AVAILABLE"
        deltas.append(after_events[key] - before_events[key])
    return sum(deltas)


def _write_memory_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    lines = [
        f"memory.current={snapshot.get('memory_current')}",
        f"memory.max={snapshot.get('memory_max')}",
    ]
    lines.extend(
        f"{key}={value}"
        for key, value in sorted(snapshot.get("memory_events", {}).items())
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_live_runtime(
    client,
    expected_sha: str,
    timeout: float = 1800.0,
) -> int:
    try:
        live_status, _live = client.get("/health/live")
        ready_status, ready = client.get("/health/ready")
        info_status, info = client.get("/info")
        runtime = info.get("runtime", {})
        if (
            live_status == 200
            and ready_status == 200
            and ready.get("ready")
            and info_status == 200
            and runtime.get("source_sha") == expected_sha
        ):
            return 0
    except Exception:
        pass

    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "stop.sh")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "start.sh")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status_code, payload = client.get("/health/ready")
            if status_code == 200 and payload.get("ready"):
                return 1
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("timed out waiting for production server readiness")


def pre_prime_authority_gate(
    client,
    expected_sha: str,
) -> dict[str, Any]:
    endpoint_results: dict[str, dict[str, Any]] = {}
    runtime: dict[str, Any] = {}
    runtime_contract: dict[str, Any] = {
        "checks": {},
        "passed": False,
        "runtime": runtime,
    }
    error = None
    try:
        for path in ("/", "/health/live", "/health/ready", "/info"):
            status_code, payload = client.get(path)
            endpoint_results[path] = {
                "http_status": status_code,
                "payload": payload,
                "pass": status_code == 200,
            }
            if path == "/health/ready":
                endpoint_results[path]["pass"] = (
                    status_code == 200 and payload.get("ready") is True
                )
            if path == "/info" and status_code == 200:
                runtime = payload.get("runtime", {})
                runtime_contract = _model_contract(runtime)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    source_sha_match = runtime.get("source_sha") == expected_sha
    checks = {
        path: bool(result.get("pass"))
        for path, result in endpoint_results.items()
    }
    checks["runtime.source_sha"] = source_sha_match
    checks["frozen_runtime_contract"] = runtime_contract["passed"] is True
    passed = not error and all(checks.values())
    failed_checks = [name for name, check in checks.items() if not check]
    if error:
        failure_reason = error
    elif failed_checks:
        labels = [
            "source SHA mismatch" if name == "runtime.source_sha" else name
            for name in failed_checks
        ]
        failure_reason = "failed authority checks: " + ", ".join(labels)
    else:
        failure_reason = None
    return {
        "passed": passed,
        "PRE_PRIME_AUTHORITY_GATE": "PASS" if passed else "FAIL",
        "GENERATION_POST_COUNT": 0,
        "generation_post_count": 0,
        "checks": checks,
        "endpoints": endpoint_results,
        "source_sha": {
            "expected": expected_sha,
            "observed": runtime.get("source_sha"),
            "match": source_sha_match,
        },
        "runtime_contract": runtime_contract["checks"],
        "failure_reason": failure_reason,
        "error": error,
    }


def run_g9(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    source_identity = getattr(args, "source_identity", None)
    if source_identity is None:
        source_identity = assert_source_identity(
            Path(getattr(args, "repo", PROJECT_ROOT)),
            args.expected_sha,
        )
    os.environ["FINAL_TPU_EXECUTION_SHA"] = args.expected_sha
    client = getattr(args, "client", None)
    model_reload_count = getattr(args, "model_reload_count", 0)
    if client is None:
        headers = {}
        if os.environ.get("API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['API_KEY']}"
        client = RestClient(
            os.environ.get("BASE_URL", "http://127.0.0.1:7860"),
            headers=headers,
            timeout=float(os.environ.get("REQUEST_TIMEOUT", "900")),
        )
        model_reload_count = _ensure_live_runtime(client, args.expected_sha)

    before_memory = read_cgroup_snapshot()
    runtime_identity_before = read_runtime_identity()
    generation_post_counter = [0]
    pre_prime_gate = pre_prime_authority_gate(client, args.expected_sha)
    endpoint_results = pre_prime_gate["endpoints"]
    if not pre_prime_gate["passed"]:
        after_memory = read_cgroup_snapshot()
        adjudication = adjudicate_g9(
            {
                "prime": False,
                "hot_1": False,
                "hot_2": False,
                "hot_cache_reuse": False,
                "compile_evidence": "FAIL",
                "text_semantic_acceptance": False,
                "vision_semantic_acceptance": False,
                "rest_acceptance": False,
                "oom_delta": _oom_delta(before_memory, after_memory),
            }
        )
        adjudication.update(
            {
                "PRE_PRIME_AUTHORITY_GATE": "FAIL",
                "GENERATION_POST_COUNT": 0,
            }
        )
        _write_json(evidence_dir / "00-context.txt", {
            "source_identity": source_identity,
            "model_reload_count": model_reload_count,
            "runtime_identity_before": runtime_identity_before,
        })
        _write_json(evidence_dir / "00-pre-prime-authority-gate.json", pre_prime_gate)
        _write_json(evidence_dir / "01-source-runtime-identity.txt", {
            "runtime_identity_before": runtime_identity_before,
            "runtime_identity_after": read_runtime_identity(),
        })
        _write_json(evidence_dir / "02-readiness.json", endpoint_results)
        _write_json(evidence_dir / "13-g9-acceptance.json", adjudication)
        (evidence_dir / "14-final-adjudication.txt").write_text(
            "\n".join(
                f"{key}={value}" for key, value in sorted(adjudication.items())
            )
            + "\n",
            encoding="utf-8",
        )
        package_evidence(evidence_dir)
        print("PRE_PRIME_AUTHORITY_GATE=FAIL")
        print("GENERATION_POST_COUNT=0")
        print(f"G9_STATUS={adjudication['G9_STATUS']}")
        return 1
    try:
        prime_payload = {
            "prompt": "hello",
            "system": "",
            "max_new_tokens": 1,
        }
        prime = _submit_and_poll(
            client, "/generate/async", prime_payload, generation_post_counter
        )
        hot1 = _submit_and_poll(
            client, "/generate/async", dict(prime_payload), generation_post_counter
        )
        hot2 = _submit_and_poll(
            client, "/generate/async", dict(prime_payload), generation_post_counter
        )
        prime_evidence = prime["result"].get("metrics", {}).get(
            "compile_cache_evidence", {}
        )
        hot1_evidence = hot1["result"].get("metrics", {}).get(
            "compile_cache_evidence", {}
        )
        hot2_evidence = hot2["result"].get("metrics", {}).get(
            "compile_cache_evidence", {}
        )
        from gemma4_server.tpu.observability import adjudicate_hot_cache

        cache_result = adjudicate_hot_cache(
            prime_evidence,
            hot1_evidence,
            hot2_evidence,
        )
        text_result = run_text_acceptance(
            client,
            evidence_dir,
            "08-text-semantic",
            generation_post_counter,
        )
        vision_result = run_vision_acceptance(client, evidence_dir, "09-vision-semantic")
        rest_acceptance = all(row["pass"] for row in endpoint_results.values()) and all(
            result["result"]["status"] == "completed"
            for result in (prime, hot1, hot2)
        )
        after_memory = read_cgroup_snapshot()
        rows = {
            "prime": _nonempty_output(prime["result"]),
            "hot_1": _nonempty_output(hot1["result"]),
            "hot_2": _nonempty_output(hot2["result"]),
            "hot_cache_reuse": cache_result["hot_cache_reuse"],
            "compile_evidence": cache_result["compile_evidence"],
            "text_semantic_acceptance": text_result["accepted"],
            "vision_semantic_acceptance": vision_result["accepted"],
            "rest_acceptance": rest_acceptance,
            "oom_delta": _oom_delta(before_memory, after_memory),
            "PRE_PRIME_AUTHORITY_GATE": "PASS",
            "GENERATION_POST_COUNT": generation_post_counter[0],
            "hot_prefill_compile_seconds": cache_result[
                "hot_prefill_compile_seconds"
            ],
            "hot_decode_compile_seconds": cache_result[
                "hot_decode_compile_seconds"
            ],
        }
    except Exception as exc:
        after_memory = read_cgroup_snapshot()
        prime = locals().get("prime", {})
        hot1 = locals().get("hot1", {})
        hot2 = locals().get("hot2", {})
        text_result = locals().get("text_result", {"accepted": False})
        vision_result = locals().get("vision_result", {"accepted": False})
        cache_result = locals().get(
            "cache_result",
            {
                "hot_cache_reuse": False,
                "compile_evidence": "FAIL",
            },
        )
        rows = {
            "prime": False,
            "hot_1": False,
            "hot_2": False,
            "hot_cache_reuse": cache_result.get("hot_cache_reuse", False),
            "compile_evidence": cache_result.get("compile_evidence", "FAIL"),
            "text_semantic_acceptance": text_result.get("accepted", False),
            "vision_semantic_acceptance": vision_result.get("accepted", False),
            "rest_acceptance": False,
            "oom_delta": _oom_delta(before_memory, after_memory),
            "error": repr(exc),
            "PRE_PRIME_AUTHORITY_GATE": "PASS",
            "GENERATION_POST_COUNT": generation_post_counter[0],
            "hot_prefill_compile_seconds": cache_result.get(
                "hot_prefill_compile_seconds"
            ),
            "hot_decode_compile_seconds": cache_result.get(
                "hot_decode_compile_seconds"
            ),
        }
    adjudication = adjudicate_g9(rows)
    _write_json(evidence_dir / "00-context.txt", {
        "source_identity": source_identity,
        "model_reload_count": model_reload_count,
        "runtime_identity_before": runtime_identity_before,
    })
    _write_json(evidence_dir / "00-pre-prime-authority-gate.json", pre_prime_gate)
    _write_json(evidence_dir / "01-source-runtime-identity.txt", {
        "runtime_identity_before": runtime_identity_before,
        "runtime_identity_after": read_runtime_identity(),
    })
    _write_json(evidence_dir / "02-readiness.json", endpoint_results)
    _write_json(evidence_dir / "03-prime-request.json", prime.get("request"))
    _write_json(evidence_dir / "04-prime-result.json", prime.get("result"))
    _write_json(evidence_dir / "05-hot1-result.json", hot1.get("result"))
    _write_json(evidence_dir / "06-hot2-result.json", hot2.get("result"))
    _write_json(evidence_dir / "07-cache-compile-evidence.json", cache_result)
    _write_json(evidence_dir / "08-text-semantic.json", text_result)
    _write_json(evidence_dir / "09-vision-semantic.json", vision_result)
    (evidence_dir / "10-rest-acceptance.txt").write_text(
        json.dumps(endpoint_results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_memory_snapshot(evidence_dir / "11-memory-before.txt", before_memory)
    _write_memory_snapshot(evidence_dir / "12-memory-after.txt", after_memory)
    _write_json(evidence_dir / "13-g9-acceptance.json", adjudication)
    (evidence_dir / "14-final-adjudication.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in sorted(adjudication.items())) + "\n",
        encoding="utf-8",
    )
    package_evidence(evidence_dir)
    print(f"G9_STATUS={adjudication['G9_STATUS']}")
    return 0 if adjudication["G9_STATUS"] == "CLOSED/PASS" else 1


def _model_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "model_preset": runtime.get("model") == "gemma4_instruct_31b",
        "backend": runtime.get("backend") == "jax",
        "jax_default_backend": runtime.get("jax_default_backend") == "tpu",
        "accelerator": runtime.get("accelerator") == "TPU v5e-8",
        "dtype": runtime.get("dtype") == "bfloat16",
        "model_class": runtime.get("model_class") == "Gemma4CausalLM",
        "backbone_class": runtime.get("backbone_class") == "Gemma4Backbone",
        "strict_weight_loading": runtime.get("strict_weight_loading") is True,
        "skip_mismatch": runtime.get("skip_mismatch") is False,
        "layout_profile": runtime.get("layout_profile")
        == "gemma4_31b_dense_candidate_a_v1",
        "checkpoint_load_strategy": runtime.get("checkpoint_load_strategy")
        == "keras_hub_native_preset_loader",
        "candidate_a_verified": runtime.get("candidate_a_verified") is True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "runtime": runtime,
    }


def adjudicate_g10(rows: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(rows)
    required = {
        "session_fresh": bool(rows.get("session_fresh", False)),
        "git_sha_exact": bool(rows.get("git_sha_exact", False)),
        "tpu_device_count": rows.get("tpu_device_count"),
        "model_load": bool(rows.get("model_load", False)),
        "text_acceptance": bool(rows.get("text_acceptance", False)),
        "vision_acceptance": bool(rows.get("vision_acceptance", False)),
        "rest_lifecycle_acceptance": bool(
            rows.get("rest_lifecycle_acceptance", False)
        ),
        "oom_delta": rows.get("oom_delta"),
        "evidence_packaged": bool(rows.get("evidence_packaged", False)),
    }
    passed = all(
        (
            required["session_fresh"],
            required["git_sha_exact"],
            required["tpu_device_count"] == 8,
            required["model_load"],
            required["text_acceptance"],
            required["vision_acceptance"],
            required["rest_lifecycle_acceptance"],
            required["oom_delta"] == 0,
            required["evidence_packaged"],
        )
    )
    normalized.update(required)
    normalized["G10_STATUS"] = "CLOSED/PASS" if passed else "OPEN/FAIL"
    normalized["G10_SESSION_FRESH"] = required["session_fresh"]
    normalized["G10_GIT_SHA_EXACT"] = required["git_sha_exact"]
    normalized["G10_TPU_DEVICE_COUNT"] = required["tpu_device_count"]
    normalized["G10_MODEL_LOAD"] = "PASS" if required["model_load"] else "FAIL"
    normalized["G10_TEXT_ACCEPTANCE"] = (
        "PASS" if required["text_acceptance"] else "FAIL"
    )
    normalized["G10_VISION_ACCEPTANCE"] = (
        "PASS" if required["vision_acceptance"] else "FAIL"
    )
    normalized["G10_REST_LIFECYCLE_ACCEPTANCE"] = (
        "PASS"
        if required["rest_lifecycle_acceptance"]
        else "FAIL"
    )
    normalized["EVIDENCE_PACKAGED"] = required["evidence_packaged"]
    return normalized


def run_g10(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    source_identity = getattr(args, "source_identity", None)
    if source_identity is None:
        source_identity = assert_source_identity(
            Path(getattr(args, "repo", PROJECT_ROOT)),
            args.expected_sha,
        )
    os.environ["FINAL_TPU_EXECUTION_SHA"] = args.expected_sha
    freshness_evidence = None
    checkpoint = getattr(args, "restart_checkpoint", None)
    if checkpoint is not None:
        session_fresh, freshness_evidence = session_fresh_from_checkpoint(
            Path(checkpoint)
        )
    else:
        session_fresh = bool(
            getattr(
                args,
                "session_fresh",
                os.environ.get("G10_SESSION_FRESH", "false").lower() == "true",
            )
        )
    client = getattr(args, "client", None)
    model_reload_count = getattr(args, "model_reload_count", 0)
    if client is None:
        headers = {}
        if os.environ.get("API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['API_KEY']}"
        client = RestClient(
            os.environ.get("BASE_URL", "http://127.0.0.1:7860"),
            headers=headers,
            timeout=float(os.environ.get("REQUEST_TIMEOUT", "900")),
        )
        model_reload_count = _ensure_live_runtime(client, args.expected_sha)

    before_memory = read_cgroup_snapshot()
    endpoint_results = {}
    model_load = {"passed": False, "checks": {}}
    text_result: dict[str, Any] = {"accepted": False}
    vision_result: dict[str, Any] = {"accepted": False}
    error = None
    try:
        for path in ("/", "/health/live", "/health/ready", "/info"):
            status_code, payload = client.get(path)
            endpoint_results[path] = {
                "http_status": status_code,
                "payload": payload,
                "pass": status_code == 200,
            }
        info_runtime = endpoint_results["/info"]["payload"].get("runtime", {})
        model_load = _model_contract(info_runtime)
        text_result = run_text_acceptance(client, evidence_dir, "06-text-acceptance")
        vision_result = run_vision_acceptance(client, evidence_dir, "07-vision-acceptance")
        rest_lifecycle = all(
            row["pass"] for row in endpoint_results.values()
        ) and text_result["accepted"] and vision_result["accepted"]
        device_count = info_runtime.get("device_count")
        if device_count is None:
            device_count = info_runtime.get("expected_tpu_devices")
        rows = {
            "session_fresh": session_fresh,
            "git_sha_exact": source_identity.get("git_sha_exact") is True
            and source_identity.get("worktree_clean") is True,
            "tpu_device_count": device_count,
            "model_load": model_load["passed"],
            "text_acceptance": text_result["accepted"],
            "vision_acceptance": vision_result["accepted"],
            "rest_lifecycle_acceptance": rest_lifecycle,
            "oom_delta": _oom_delta(before_memory, read_cgroup_snapshot()),
        }
    except Exception as exc:
        error = repr(exc)
        rows = {
            "session_fresh": session_fresh,
            "git_sha_exact": source_identity.get("git_sha_exact") is True
            and source_identity.get("worktree_clean") is True,
            "tpu_device_count": None,
            "model_load": False,
            "text_acceptance": text_result["accepted"],
            "vision_acceptance": vision_result["accepted"],
            "rest_lifecycle_acceptance": False,
            "oom_delta": _oom_delta(before_memory, read_cgroup_snapshot()),
        }
    after_memory = read_cgroup_snapshot()
    rows["oom_delta"] = _oom_delta(before_memory, after_memory)
    rows["evidence_packaged"] = True
    rows["error"] = error
    adjudication = adjudicate_g10(rows)
    _write_json(
        evidence_dir / "00-context.txt",
        {
            "source_identity": source_identity,
            "session_fresh": session_fresh,
            "model_reload_count": model_reload_count,
            "freshness_evidence": freshness_evidence,
        },
    )
    _write_json(
        evidence_dir / "01-fresh-session-identity.txt",
        freshness_evidence or read_runtime_identity(),
    )
    _write_json(evidence_dir / "02-git-provenance.txt", source_identity)
    _write_json(
        evidence_dir / "03-tpu-topology.txt",
        {
            "device_count": rows.get("tpu_device_count"),
            "runtime": endpoint_results.get("/info", {}).get("payload", {}).get("runtime", {}),
        },
    )
    _write_json(evidence_dir / "04-model-load.json", model_load)
    _write_json(
        evidence_dir / "05-sharding.json",
        {
            "candidate_a_verified": model_load.get("runtime", {}).get(
                "candidate_a_verified"
            ),
            "layout_profile": model_load.get("runtime", {}).get("layout_profile"),
        },
    )
    _write_json(evidence_dir / "06-text-acceptance.json", text_result)
    _write_json(evidence_dir / "07-vision-acceptance.json", vision_result)
    (evidence_dir / "08-rest-lifecycle.txt").write_text(
        json.dumps(endpoint_results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_memory_snapshot(evidence_dir / "09-memory-before.txt", before_memory)
    _write_memory_snapshot(evidence_dir / "10-memory-after.txt", after_memory)
    _write_json(evidence_dir / "11-g10-acceptance.json", adjudication)
    (evidence_dir / "12-final-adjudication.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in sorted(adjudication.items())) + "\n",
        encoding="utf-8",
    )
    package_evidence(evidence_dir)
    print(f"G10_STATUS={adjudication['G10_STATUS']}")
    return 0 if adjudication["G10_STATUS"] == "CLOSED/PASS" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g9", "g10"), required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--restart-checkpoint", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "g9":
        return run_g9(args)
    return run_g10(args)


if __name__ == "__main__":
    raise SystemExit(main())
