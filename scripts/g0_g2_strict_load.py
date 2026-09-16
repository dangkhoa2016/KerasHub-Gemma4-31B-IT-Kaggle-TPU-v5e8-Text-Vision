#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path

from gemma4_server.core.memory import (
    cgroup_snapshot
)
from gemma4_server.tpu.distribution import (
    build_distribution
)
from gemma4_server.tpu.engine import (
    Gemma4TPUEngine
)

def main():
    import jax, keras

    path = os.environ["MODEL_PATH"]
    expected = int(
        os.environ.get(
            "EXPECTED_TPU_DEVICES","8"
        )
    )
    devices = list(jax.devices("tpu"))
    if len(devices) != expected:
        raise RuntimeError(
            f"Expected {expected} TPU devices, "
            f"found {len(devices)}"
        )

    shape = tuple(
        int(x)
        for x in os.environ.get(
            "MESH_SHAPE","1,8"
        ).split(",")
    )
    axes = tuple(
        x.strip()
        for x in os.environ.get(
            "MESH_AXIS_NAMES",
            "batch,model",
        ).split(",")
    )

    _mesh, _layout, dist = (
        build_distribution(
            keras,
            jax,
            shape=shape,
            axis_names=axes,
        )
    )

    engine = Gemma4TPUEngine(
        path,
        os.environ.get(
            "MODEL_DTYPE","bfloat16"
        ),
        dist,
        vision_enabled=True,
        generation_length_buckets=(
            512,768,1024,1536,2048
        ),
        max_generation_length=int(
            os.environ.get(
                "MAX_GENERATION_LENGTH","2048"
            )
        ),
        min_sharded_parameter_percent=float(
            os.environ.get(
                "MIN_SHARDED_PARAMETER_PERCENT",
                "80",
            )
        ),
    )

    out = Path(
        "artifacts/g0-g2/strict-load.json"
    )
    out.parent.mkdir(
        parents=True, exist_ok=True
    )

    try:
        metadata = engine.load()
        result = {
            "status":"strict_load_pass",
            "generation_executed":False,
            "mesh_shape":list(shape),
            "mesh_axis_names":list(axes),
            "memory":cgroup_snapshot(),
            **metadata,
        }
        guard=float(
            os.environ.get(
                "MEMORY_GUARD_GIB","300"
            )
        )
        current=result["memory"]["current_gib"]
        if (
            current is not None
            and current >= guard
        ):
            raise RuntimeError(
                f"Memory guard failed: "
                f"{current} >= {guard}"
            )
    except Exception as exc:
        result={
            "status":"strict_load_failed",
            "generation_executed":False,
            "error_type":type(exc).__name__,
            "error":str(exc),
            "memory":cgroup_snapshot(),
        }
        out.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ))
        raise

    out.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        result,
        indent=2,
        sort_keys=True,
    ))
    print("G2_STRICT_LOAD_PASS")

if __name__=="__main__":
    main()
