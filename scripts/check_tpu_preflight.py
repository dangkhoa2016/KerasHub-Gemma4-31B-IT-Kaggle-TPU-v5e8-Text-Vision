#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, sys

child = r"""
import os
os.environ.setdefault("KERAS_BACKEND","jax")
os.environ.setdefault("JAX_PLATFORMS","tpu,cpu")
import jax
devices=jax.devices("tpu")
expected=int(os.environ.get("EXPECTED_TPU_DEVICES","8"))
print("JAX:",jax.__version__)
print("TPU devices:",devices)
if len(devices)!=expected:
    raise SystemExit(
        f"Expected {expected} TPU devices, found {len(devices)}"
    )
"""

p = subprocess.run(
    [sys.executable,"-c",child],
    text=True,
    capture_output=True,
    env=os.environ.copy(),
)
print(p.stdout, end="")
if p.stderr:
    print(p.stderr, file=sys.stderr, end="")
raise SystemExit(p.returncode)
