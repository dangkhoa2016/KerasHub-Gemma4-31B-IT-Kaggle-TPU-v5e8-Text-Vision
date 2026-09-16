# G8 Live Authority Corrective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close G8 only after one uninterrupted real Gemma4 31B TPU v5e-8 lifecycle proves initial load/compile, async inference, canonical restart, sequential replacement load, and post-restart inference.

**Architecture:** Preserve the existing CPU-qualified source and historical archives. Capture a fresh live-only evidence directory around one server process and one model worker, with model A and model B owned sequentially in the same TPU allocation; package only live evidence plus references to prior authority.

**Tech Stack:** Python 3.12, JAX/Keras/KerasHub, Flask/Waitress REST server, Kaggle TPU v5e-8, shell evidence capture, tar/gzip, SHA-256.

**Spec:** `/kaggle/working/gemma4-31b-g8-live-authority-only-contiguous-tpu-corrective-codex-directive-2026-09-14.md`

## Global Constraints

- `TPU_ALLOCATION_COUNT_TARGET=1`.
- `TPU_RECONFIGURATION_DURING_G8=FORBIDDEN`.
- `MODEL_INSTANCE_ACTIVE_MAX=1`, `MODEL_LOAD_CONCURRENCY_MAX=1`, `MODEL_WORKER_COUNT=1`, `SERVER_PROCESS_COUNT=1`.
- Preserve the prior G8 CPU archive and G7 archive; do not rerun G3-G7 or G9.
- Use `model.compile(sampler="greedy", run_eagerly=True)` and Candidate-A sharding exactly as frozen.
- Stop on any live failure; no automatic second TPU allocation, patch, or rerun.
- Do not release/reconfigure TPU or switch to CPU after the live phase begins.
- `TAG=false`, `RELEASE=false`, and `G9_STARTED=false`.

---

### Task 1: Minimal preflight and source-continuity record

**Files:**
- Read: canonical repository under `/kaggle/working/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision`.
- Read: prior G8 CPU and G7 authority archives.
- Create: fresh live evidence directory under `/kaggle/working/gemma4-31b-g8-live-authority-<UTCSTAMP>/`.

- [x] Verify repository/files, prior archive hashes, G7 hash, runtime identity, and source hashes.
- [x] Compare CPU-qualified manager/worker/store files against the prior G8 archive and stop on unexplained drift.
- [x] Run only compileall, shell syntax, JSON parse, server import, and focused authority-harness checks without model loading.
- [x] Record corrective context, references, hashes, runtime versions, and pre-live memory/OOM state.

### Task 2: TPU gate and initial real model lifecycle

**Files:**
- Create: live hardware, server stdout/stderr, memory, and initial lifecycle/readiness evidence files.
- Read: canonical server/worker/TPU implementation without modifying it.

- [ ] Confirm eight visible JAX TPU devices and accepted device exposure. **BLOCKED:** no TPU devices or `libtpu.so` are exposed in the current session; stop before model load.
- [ ] Start exactly one server process and one canonical model worker in the same allocation.
- [ ] Load the frozen Gemma4 31B checkpoint, run explicit greedy eager compile, verify Candidate-A sharding, and reach canonical ready state.
- [ ] Record timings, process/worker counters, and cgroup memory/OOM snapshots.

### Task 3: Initial async request and canonical restart

**Files:**
- Create: REST request/poll, restart, lifecycle, and memory evidence files.
- Read: canonical REST endpoint/schema and lifecycle responses.

- [ ] Submit exactly one minimal model-backed async REST request and poll to canonical success.
- [ ] Submit exactly one canonical restart request and prove readiness drops, old worker stops/joins, and old generation cannot reopen readiness.
- [ ] Prove replacement ownership begins only after old model ownership is inactive.

### Task 4: Sequential replacement lifecycle and post-restart proof

**Files:**
- Create: replacement lifecycle/readiness/request/result/final memory evidence files.

- [ ] Load replacement Gemma4 31B in the same TPU allocation with no overlap.
- [ ] Run eager compile and Candidate-A verification, reach ready, and submit exactly one post-restart model-backed async REST request.
- [ ] Record final OOM delta and lifecycle counters; stop only application/model lifecycle if needed while preserving TPU allocation.

### Task 5: Adjudication and evidence package

**Files:**
- Create: `22-g8-live-authority.json`, `23-final-adjudication.txt`, `SHA256SUMS`, and archive sidecar.
- Preserve: prior CPU-only G8 adjudication and G7 evidence.

- [ ] Validate every directive acceptance field and set G8 pass/closeout only if all hard gates pass.
- [ ] Package compact live corrective evidence with references to prior archives, verify archive and sidecar hashes.
- [ ] Print the required final report with `G9_ENTRY_ELIGIBLE=true` only on live pass, `G9_STARTED=false`, `TPU_RELEASED_AFTER_G8=false`, `TAG=false`, and `RELEASE=false`.
