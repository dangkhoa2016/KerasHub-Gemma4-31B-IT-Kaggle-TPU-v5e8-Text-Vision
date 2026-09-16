#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, urllib.request

url=os.environ.get(
    "READY_URL",
    "http://127.0.0.1:7860/health/ready",
)
timeout=float(
    os.environ.get("READY_TIMEOUT","1800")
)
deadline=time.monotonic()+timeout

while time.monotonic()<deadline:
    try:
        with urllib.request.urlopen(
            url, timeout=5
        ) as response:
            data=json.loads(
                response.read().decode()
            )
            print(json.dumps(
                data, indent=2
            ))
            if (
                response.status==200
                and data.get("ready")
            ):
                print("READY")
                raise SystemExit(0)
    except Exception as exc:
        print(
            f"waiting: "
            f"{type(exc).__name__}: {exc}"
        )
    time.sleep(5)

raise SystemExit(
    "Timed out waiting for ready"
)
