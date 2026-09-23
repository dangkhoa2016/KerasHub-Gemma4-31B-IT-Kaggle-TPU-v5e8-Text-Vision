#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
    ) | {"pid": str(os.getpid())}


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


if __name__ == "__main__":
    raise SystemExit("orchestration modes are added in a later task")
