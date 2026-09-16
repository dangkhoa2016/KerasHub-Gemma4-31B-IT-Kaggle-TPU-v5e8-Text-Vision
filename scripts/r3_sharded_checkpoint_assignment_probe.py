from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metadata_from_event(event, prefix):
    return {
        key[len(prefix) :]: value
        for key, value in event.items()
        if key.startswith(prefix)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import keras

    from gemma4_server.tpu.distribution import build_distribution
    from gemma4_server.tpu.engine import Gemma4TPUEngine

    timeline = []

    def on_event(event, payload):
        timeline.append({
            "event": event,
            "monotonic_seconds": time.monotonic(),
            **payload,
        })

    devices = list(jax.devices("tpu"))
    if len(devices) != 8:
        raise RuntimeError(f"Expected 8 TPU devices, found {len(devices)}")

    mesh, _layout_map, distribution = build_distribution(
        keras,
        jax,
        shape=(1, 8),
        axis_names=("batch", "model"),
        data_axis="batch",
        model_axis="model",
    )

    print("R3_PROBE_START")
    engine = Gemma4TPUEngine(
        str(Path(args.model_path)),
        "bfloat16",
        distribution,
        r3_event_callback=on_event,
    )
    metadata = engine.load()

    token_events = [
        event
        for event in timeline
        if event.get("target_path", "").endswith(
            "token_embedding/embeddings"
        )
    ]
    required_events = [
        "R3_TARGET_ENTER",
        "SOURCE_READY",
        "TARGET_SHARDING_OBSERVED",
        "SHARDED_TRANSFER_BEGIN",
        "SHARD_CALLBACK_ENTER",
        "SHARD_CALLBACK_RETURN",
        "SHARDED_VALUE_CREATED",
        "SHARDED_VALUE_SHARDING_VERIFIED",
        "ORIGINAL_ASSIGN_WITH_SHARDED_VALUE_ENTER",
        "ORIGINAL_ASSIGN_WITH_SHARDED_VALUE_SUCCESS",
        "TARGET_SHARDING_AFTER_VERIFIED",
        "R3_SAMPLE_VALUE_CHECK_PASS",
    ]
    observed_events = [event["event"] for event in token_events]
    for required in required_events:
        if required not in observed_events:
            raise RuntimeError(
                f"Missing token embedding R3 event: {required}"
            )

    before = next(
        event for event in token_events
        if event["event"] == "TARGET_SHARDING_OBSERVED"
    )
    source = next(
        event for event in token_events
        if event["event"] == "SOURCE_READY"
    )
    sharded = next(
        event for event in token_events
        if event["event"] == "SHARDED_VALUE_SHARDING_VERIFIED"
    )
    after = next(
        event for event in token_events
        if event["event"] == "TARGET_SHARDING_AFTER_VERIFIED"
    )
    sample = next(
        event for event in token_events
        if event["event"] == "R3_SAMPLE_VALUE_CHECK_PASS"
    )

    target_weight = next(
        weight
        for weight in engine.model.weights
        if str(getattr(weight, "path", "")).endswith(
            "token_embedding/embeddings"
        )
    )
    target_value = target_weight.value
    target_shards = list(target_value.addressable_shards)
    target_shard_shapes = [
        [int(dimension) for dimension in shard.data.shape]
        for shard in target_shards
    ]

    _write_json(evidence_dir / "r3-timeline-events.json", timeline)
    _write_json(evidence_dir / "r3-target-metadata-before.json", {
        "TARGET_VALUE_CLASS": before["target_value_class"],
        "TARGET_VALUE_SHARDING_CLASS": before["target_sharding_class"],
        "TARGET_VALUE_SHARDING_REPR": before["target_sharding_repr"],
        "TARGET_VALUE_SHARDING_SPEC": before["target_sharding_spec"],
        "TARGET_VALUE_SHARDING_MESH": before["target_sharding_mesh"],
        "TARGET_ADDRESSABLE_SHARD_COUNT": before[
            "target_addressable_shard_count"
        ],
        "TARGET_SHARD_SHAPES": before["target_shard_shapes"],
        "MESH": repr(mesh),
    })
    _write_json(evidence_dir / "r3-source-metadata.json", {
        "SOURCE_CLASS": source["source_class"],
        "SOURCE_SHAPE": source["source_shape"],
        "SOURCE_DTYPE": source["source_dtype"],
        "SOURCE_NBYTES": source["source_nbytes"],
        "SOURCE_SHARDING": None,
    })
    _write_json(evidence_dir / "r3-sharded-value-metadata.json", {
        "SHARDED_VALUE_CLASS": sharded["sharded_value_class"],
        "SHARDED_VALUE_SHARDING_CLASS": sharded[
            "sharded_sharding_class"
        ],
        "SHARDED_VALUE_SHARDING": sharded["sharded_sharding_repr"],
        "SHARDED_VALUE_SPEC": sharded["sharded_sharding_spec"],
        "SHARDED_VALUE_MESH": sharded["sharded_sharding_mesh"],
        "SHARDED_VALUE_ADDRESSABLE_SHARDS": sharded[
            "sharded_addressable_shard_count"
        ],
        "SHARDED_VALUE_SHARD_SHAPES": sharded["sharded_shard_shapes"],
    })
    _write_json(evidence_dir / "r3-target-metadata-after.json", {
        "TARGET_VALUE_CLASS": after["target_value_class"],
        "TARGET_VALUE_SHARDING_CLASS": after["target_sharding_class"],
        "TARGET_VALUE_SHARDING": after["target_sharding_repr"],
        "TARGET_VALUE_SHARDING_SPEC": after["target_sharding_spec"],
        "TARGET_VALUE_SHARDING_MESH": after["target_sharding_mesh"],
        "TARGET_ADDRESSABLE_SHARD_COUNT": after[
            "target_addressable_shard_count"
        ],
        "TARGET_SHARD_SHAPES": target_shard_shapes,
    })
    _write_json(evidence_dir / "r3-sample-value-check.json", {
        "R3_SAMPLE_VALUE_CHECK": "PASS",
        "event": sample,
        "rows": [
            0, 32767, 32768, 65535, 65536, 98303, 98304, 131071,
            131072, 163839, 163840, 196607, 196608, 229375, 229376,
            262143,
        ],
        "columns": [0, 1, 5375],
        "full_embedding_downloaded": False,
    })
    _write_json(evidence_dir / "r3-model-load-metadata.json", metadata)

    print("R3_ASSIGN_RESULT=SUCCESS")
    print("R3_SAMPLE_VALUE_CHECK=PASS")
    print("R3_NARROW_PROOF=PASS")
    print("R3_DIAGNOSTIC_STOP")


if __name__ == "__main__":
    main()
