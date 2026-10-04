#!/usr/bin/env python3
from __future__ import annotations
import importlib.metadata as md
import os, subprocess, sys

expected = os.environ.get(
    "EXPECTED_LIBTPU_VERSION", "0.0.49"
).strip()
mode = os.environ.get(
    "INSTALL_LIBTPU_IF_MISSING", "auto"
).strip().lower()

try:
    current = md.version("libtpu")
except md.PackageNotFoundError:
    current = None

if current == expected:
    print(f"[libtpu] expected runtime already installed: {current}")
    raise SystemExit(0)

if mode in {"false","0","no"}:
    state = f"installed={current}" if current else "missing"
    raise SystemExit(
        f"libtpu {state}; expected={expected}; installation disabled"
    )

action = "upgrading" if current else "installing"
print(
    f"[libtpu] {action} to {expected} without dependencies"
)
subprocess.run([
    sys.executable,
    "-m","pip","install",
    "--disable-pip-version-check",
    "--root-user-action=ignore",
    "--no-warn-conflicts",
    "--no-deps",
    "--progress-bar","off",
    f"libtpu=={expected}",
], check=True)
