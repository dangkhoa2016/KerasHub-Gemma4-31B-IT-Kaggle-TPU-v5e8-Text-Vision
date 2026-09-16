# G3 Generation Architecture Corrective vNext Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the G3 authority generation use the exact one-token native KerasHub shape, release avoidable post-load host memory, persist load/generation evidence before a possible SIGKILL, and run the authorized TPU adjudication.

**Architecture:** Keep `plan_generation()` and configured production buckets unchanged. Add a project-owned `plan_authority_generation()` and `Gemma4TPUEngine.generate_text_authority()` that call the verified native API with `max_length=prompt_tokens+1`, `strip_prompt=True`, and no warmup. After successful load and Candidate-A verification, run guarded `gc.collect()`/`malloc_trim(0)` and expose RSS/cgroup measurements. The outer authority harness writes fsync'd marker files at the two durability boundaries and packages the one TPU result.

**Tech Stack:** Python 3, unittest, Keras 3/KerasHub native CausalLM generation, JAX TPU v5e-8, standard-library cgroup/RSS telemetry.

**Spec:** `/kaggle/working/gemma4-31b-g3-generation-architecture-corrective-vnext-tpu-authority-directive-2026-09-12.md`

## Global Constraints

- Preserve checkpoint, model revision, BF16, mesh `[1,8]`, Candidate-A sharding, R3 assignment, and 8-device TPU execution.
- Keep production REST bucket policy unchanged; only the narrow G3 authority path may bypass it.
- Use exactly one model load, one generation call, no warmup, no benchmark, no REST, no vision, and no G4.
- Do not edit site-packages, use `skip_mismatch`, mutate checkpoints, inject weights manually, change precision, or use CPU fallback.
- Treat native `model.generate(max_length=...)` as the selected path only because installed source inspection verified exact sequence-length propagation.

### Task 1: Add failing CPU architecture and cleanup tests

**Files:**
- Modify: `tests/test_generation.py`
- Modify: `tests/test_source_contract.py`
- Create: `tests/test_g3_generation_architecture_corrective.py`

- [ ] Add red tests for exact authority planning, unchanged production bucket planning, one-token call arguments, post-load cleanup ordering/guarding, marker content/fsync, and invariant strings.
- [ ] Run the focused tests and confirm they fail for missing vNext behavior.

### Task 2: Implement project generation and cleanup behavior

**Files:**
- Modify: `src/gemma4_server/tpu/generation.py`
- Modify: `src/gemma4_server/tpu/engine.py`

- [ ] Add exact authority planning without changing `plan_generation()`.
- [ ] Add guarded memory measurement and allocator trim; call cleanup only after successful load/sharding verification and before generation.
- [ ] Add `generate_text_authority()` with one native `model.generate()` call at exact length and explicit authority metadata; leave normal `generate_text()` bucketed.
- [ ] Run focused tests and the full project CPU suite.

### Task 3: Update the one-process authority harness and evidence

**Files:**
- Modify: `/kaggle/working/gemma4-g3-minimum-shape-authority.py`
- Modify: `/kaggle/working/test_gemma4_g3_minimum_shape_authority.py`

- [ ] Switch preflight proof to exact authority length 11 while recording current production buckets separately.
- [ ] Write and fsync durable model/sharding markers after load and generation-start markers immediately before the sole call.
- [ ] Record cleanup telemetry, exact path identity, model/generation counters, cgroup/RSS fields, and vNext adjudication names.
- [ ] Add/run CPU harness tests without importing JAX/Keras in preflight.

### Task 4: Run CPU gates, freeze source, and execute one TPU authority

**Files:**
- Create: `/kaggle/working/gemma4-31b-g3-generation-architecture-corrective-vnext-evidence-<timestamp>/`

- [ ] Run `pip check`, project tests, corrective tests, `py_compile`, forbidden-pattern scan, and model/weight-map preflight.
- [ ] Freeze source snapshot and launch exactly one TPU authority process after all CPU gates pass.
- [ ] Capture before/after cgroup memory evidence, telemetry, durable markers, exit status, and result JSON.

### Task 5: Final verification and report

- [ ] Verify checksums/archive and report the required corrective path, KerasHub source inspection, cleanup measurements, TPU result, cgroup deltas, and G3/G4 adjudication.
- [ ] Stop after G3 PASS or directly adjudicated new-architecture OOM; never start G4.
