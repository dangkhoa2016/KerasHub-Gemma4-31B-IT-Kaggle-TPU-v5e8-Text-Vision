#!/usr/bin/env python3
from __future__ import annotations
import importlib.metadata as md
import os, subprocess, sys

expected = os.environ.get(
    "EXPECTED_LIBTPU_VERSION", "0.0.17"
).strip()
mode = os.environ.get(
    "INSTALL_LIBTPU_IF_MISSING", "auto"
).strip().lower()

try:
    current = md.version("libtpu")
except md.PackageNotFoundError:
    current = None

if current:
    print(f"[libtpu] existing runtime retained: {current}")
    if current != expected:
        print(
            f"[libtpu] advisory: reference={expected}, "
            f"existing={current}"
        )
    raise SystemExit(0)

if mode in {"false","0","no"}:
    raise SystemExit(
        "libtpu missing and installation disabled"
    )

print(
    f"[libtpu] installing {expected} without dependencies"
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
