# G9 PRIME/HOT Acceptance Runbook

This runbook is prepared during the final CPU phase and must not be executed
until the final TPU one-shot. It consumes the exact frozen Git SHA and the
existing TPU allocation.

## Frozen runtime contract

```text
MODEL_PRESET=gemma4_instruct_31b
KERAS_BACKEND=jax
MODEL_DTYPE=bfloat16
TPU_DEVICE_COUNT=8
MESH_SHAPE=[1,8]
MESH_AXES=[batch,model]
layout_profile=gemma4_31b_dense_candidate_a_v1
checkpoint_load_strategy=keras_hub_native_preset_loader
production buckets=(16,512,768,1024,1536,2048)
```

Do not change Candidate-A sharding, bucket policy, loader semantics, or model
configuration while running G9.

## Execution order

1. Verify the checkout is at `FINAL_CPU_FROZEN_SHA`, the tree is clean, and
   the eight-device TPU v5e-8 allocation is available.
2. Run one PRIME request through the production path and capture readiness,
   bucket selection, compile events, generation result, memory/cgroup state,
   and OOM counters.
3. Run HOT-1 and HOT-2 sequentially without reloading the model. Record cache
   reuse from actual engine metrics/logs.
4. Repeat semantic text acceptance and the existing vision acceptance using
   the repository's production request contract.
5. Validate the production REST/lifecycle endpoints and package only compact
   evidence; never include weights, caches, credentials, PID/state files, or
   private logs.

## Required adjudication

```text
PRIME=PASS
HOT_1=PASS
HOT_2=PASS
HOT_CACHE_REUSE=true
HOT_PREFILL_COMPILE_SECONDS=0
HOT_DECODE_COMPILE_SECONDS=0
TEXT_SEMANTIC_ACCEPTANCE=PASS
VISION_SEMANTIC_ACCEPTANCE=PASS
OOM_DELTA=0
```

Use only metrics emitted by the engine/server. If an engine metric is not
available, record `NOT_AVAILABLE`; do not fabricate zeroes or infer cache
reuse from elapsed time alone. G9 remains failed unless all required semantic
and memory conditions are directly evidenced.
