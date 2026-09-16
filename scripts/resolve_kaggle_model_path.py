#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from gemma4_server.core.model_path import (
    resolve_model_path
)

p=argparse.ArgumentParser()
p.add_argument("--preferred",default="")
p.add_argument("--input-root",default="/kaggle/input")
p.add_argument(
    "--preset-name",
    default="gemma4_instruct_31b",
)
a=p.parse_args()
preferred=(
    Path(a.preferred)
    if a.preferred.strip()
    else None
)
print(resolve_model_path(
    preferred,
    Path(a.input_root),
    a.preset_name,
))
