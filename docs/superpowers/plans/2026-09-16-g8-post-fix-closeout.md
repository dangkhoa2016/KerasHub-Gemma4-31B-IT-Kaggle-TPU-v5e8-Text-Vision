# G8 Post-Fix Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a truthful G8 post-fix closeout sidecar from the frozen successful TPU authority evidence and fresh post-patch static regression.

**Architecture:** Treat `/kaggle/working/aaaa.md` as the controlling directive and preserve all prior evidence packages. Verify the scoped collector fallback and its regression test, run the repository-native complete CPU/static workflow, then write a compact sidecar that records the frozen live facts, exact source hashes, historical classification, and any evidence paths without creating or rerunning TPU work.

**Tech Stack:** Python 3.12, unittest, compileall, Bash syntax checks, SHA-256, JSON/text evidence files.

**Spec:** `/kaggle/working/aaaa.md`

## Global Constraints

- Do not send another generation request, reload Gemma4, restart the TPU worker for evidence, start a new Kaggle session, start G9, create a tag, or create a release.
- Preserve the collector corrective exactly at `src/gemma4_server/workers/manager.py` and its nested `metrics.generation_seconds` regression test.
- Use the established canonical workflow `bash scripts/test_unit.sh`; the full suite and compileall must pass.
- Preserve runtime terminology exactly, including `layout_profile`, `checkpoint_load_strategy`, `generation_mode`, and `runtime_validation`.
- Preserve the initial fresh-allocation gate failure and later same-session corrective recovery as historical sequence.
- Set `G8_FINAL_CLOSEOUT_ELIGIBLE=true` only when frozen live authority, fresh full regression, compileall, and provenance all pass; keep `G9_STARTED=false`, `TAG=false`, and `RELEASE=false`.

### Task 1: Verify corrective scope and immutable authority inputs

**Files:**
- Read: `/kaggle/working/aaaa.md`
- Read: frozen authority files under `/kaggle/working/temp` and `/kaggle/working/gemma4-31b-g8-*`
- Read: `src/gemma4_server/workers/manager.py`
- Read: `tests/test_g8_manager_lifecycle.py`

- [x] **Step 1: Confirm the production corrective and regression test.**

  Verify that `job_completed` derives `metrics = message.get("metrics") or {}` and passes `message.get("inference_seconds", metrics.get("generation_seconds"))` to `mark_completed`, and that the lifecycle test sends only nested `metrics.generation_seconds`.

- [x] **Step 2: Verify frozen live facts without invoking TPU code.**

  Cross-check the final job ID, status, timings, bucket flags, runtime facts, and OOM deltas against the controlling directive and available authority logs. Treat unavailable result payload contents as unavailable; do not reconstruct them.

- [x] **Step 3: Verify historical evidence is preserved.**

  Confirm that failed attempts and prior G8/G7 archives remain present and that no historical closeout is rewritten.

### Task 2: Run fresh post-fix canonical CPU/static regression

**Files:**
- Execute: `scripts/test_unit.sh`
- Read: `tests/`
- Read: `src/`, `scripts/`, `clients/python/`

- [x] **Step 1: Run the established complete workflow.**

  Run `bash scripts/test_unit.sh` from the repository root. This must run unittest discovery, compileall, and the project’s established shell syntax checks. Record the exact test count and exit status.

- [x] **Step 2: Run the scoped collector test explicitly.**

  Run `python3 -m unittest tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_job_completed_event_uses_metrics_when_inference_seconds_is_nested` and require a zero exit status.

- [x] **Step 3: Stop on any failure.**

  If either command fails, do not claim eligibility, do not rerun TPU work, and diagnose only the failing tests.

### Task 3: Create post-fix evidence sidecar and provenance

**Files:**
- Create: `artifacts/g8/post-fix-closeout-20260916/README.md`
- Create: `artifacts/g8/post-fix-closeout-20260916/final-adjudication.txt`
- Create: `artifacts/g8/post-fix-closeout-20260916/SHA256SUMS`

- [x] **Step 1: Record exact source hashes.**

  Record SHA256 for `src/gemma4_server/workers/manager.py` and `tests/test_g8_manager_lifecycle.py`, plus the exact corrective excerpt and test name.

- [x] **Step 2: Record authority and runtime metadata.**

  Include the frozen final job facts, JAX facts, runtime terminology, readiness/result/log source paths, OOM deltas, and an explicit note for any authority payload not present as a local file.

- [x] **Step 3: Record truthful adjudication markers.**

  Set the required final markers only if all gates passed. Keep the historical failed initial gate classification and set `G9_STARTED=false`, `TAG=false`, and `RELEASE=false`.

- [x] **Step 4: Hash the sidecar files.**

  Generate `SHA256SUMS` only after the sidecar contents are final and verify it with `sha256sum -c`.

### Task 4: Final verification and handoff

**Files:**
- Read: `artifacts/g8/post-fix-closeout-20260916/`

- [x] **Step 1: Re-run sidecar checksum verification and inspect markers.**

- [x] **Step 2: Confirm no forbidden live operation or G9/tag/release marker occurred.**

- [x] **Step 3: Report the required directive output block with paths and test count.**
