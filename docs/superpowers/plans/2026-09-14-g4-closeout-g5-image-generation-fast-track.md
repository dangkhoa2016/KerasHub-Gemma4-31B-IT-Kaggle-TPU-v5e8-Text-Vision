# G4 Closeout and G5 Image-Conditioned Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the authoritative G4 PASS evidence without another TPU run, then prove one-token Gemma4 31B image-conditioned generation on the existing Candidate-A TPU architecture within the directive's two-attempt budget.

**Architecture:** Preserve the G3/G4 authority archives as immutable inputs. Create a compact G4 freeze manifest/closeout package from the existing evidence, then add a G5-specific CPU-discoverable vision adapter and one-process authority runner that reuses the existing `Gemma4TPUEngine` loader, BF16, `[1,8]` mesh, and Candidate-A. Prefer split/cache vision only when the installed API directly supports it; otherwise use the installed native multimodal path and record that choice.

**Tech Stack:** Python 3.12, Keras 3.15.0, KerasHub 0.29.1, Keras-NLP 0.29.1, JAX 0.10.2, JAXlib 0.10.2, NumPy 2.5.0, libtpu 0.0.17, Pillow, Bash, unittest.

**Spec:** `/kaggle/working/gemma4-31b-g4-freeze-closeout-to-g5-image-generation-fast-track-codex-directive-2026-09-14.md`

## Global Constraints

- Do not rerun G3, rerun G4, change the G3 freeze artifact, mutate either authority archive, tag, release, push, or change model/checkpoint/BF16/mesh/Candidate-A/R3 science.
- G4 freeze verification is CPU/static-only and must not initialize JAX or execute a model.
- Reuse `gemma4_instruct_31b`, the installed checkpoint path, BF16, mesh `[1,8]`, axes `[batch,model]`, Candidate-A, and one model load per G5 attempt.
- Use a tiny deterministic RGB fixture, one short prompt, one generated token, one generation call, and no duplicate model/image tensors.
- G5 allows one primary real TPU attempt plus one corrective rerun only after a direct root cause and passing CPU/static tests; no third attempt.
- Preserve historical source hashes exactly; later edits are recorded as `POST_G4_SOURCE_CHANGE` and do not change `artifacts/g4/g4-pass-freeze.json`.

---

### Task 1: Freeze and verify G4 evidence

**Files:**
- Create: `artifacts/g4/g4-pass-freeze.json`
- Create: `docs/G4-CLOSEOUT.md`
- Create: `docs/G4-TROUBLESHOOTING.md`
- Modify: `DEVELOPMENT-STATUS.md`, `docs/ROADMAP.md`, `docs/ROADMAP.vi.md`
- Create: `/kaggle/working/gemma4-31b-g4-pass-freeze-closeout-<UTC>/...`

**Interfaces:**
- Consumes the verified G4 authority archive, its evidence directory, the frozen G3 archive, and the exact `02-g4-source-hashes-after.txt` contents.
- Produces an immutable G4 manifest, closeout/troubleshooting docs, compact archive, sidecar checksum, and status markers with `G5_STARTED=false`.

- [ ] Verify sidecar checksum, internal `SHA256SUMS`, required JSON fields, comparison/adjudication agreement, G3 archive hash, and current source hashes.
- [ ] Write the G4 manifest using the historical hash map verbatim, never recomputed as historical evidence.
- [ ] Write closeout and troubleshooting documents with the exact PASS/CLOSED/eligibility markers and limitations.
- [ ] Build the compact evidence package with only the ten listed files plus `SHA256SUMS`; archive and checksum it.
- [ ] Run CPU/static verification and confirm `G4_FREEZE_VERIFICATION=PASS`, `G4_IMMUTABLE=true`, `G4_RERUN_FORBIDDEN=true`.

### Task 2: CPU-first G5 API discovery and contract tests

**Files:**
- Create: `artifacts/g5/g5-api-discovery.md`
- Create: `tests/test_g5_contract.py`
- Inspect only: installed Gemma4/KerasHub multimodal preprocessor and model source.

**Interfaces:**
- Produces an observed/inferred/unsupported API record and testable contracts for fixture loading, preprocessing, vision-conditioning detection, single-load authority ordering, and one-token output validation.

- [ ] Inspect the installed preprocessor signature/source and the existing repository image/vision flow, recording exact accepted image structure and result keys.
- [ ] Write failing tests for the discovered contract before adding G5 implementation code.
- [ ] Run the focused tests and confirm they fail for missing G5 modules or missing contract implementation.
- [ ] Use an existing fixture if available; otherwise define a deterministic 64x64 RGB PNG fixture contract without network access.

### Task 3: Implement the minimal G5 vision helper and authority runner

**Files:**
- Create: `src/gemma4_server/tpu/g5_vision.py`
- Create: `scripts/g5_image_authority.py`
- Create: `scripts/run_g5_image_authority.sh`
- Modify: `tests/test_g5_contract.py`

**Interfaces:**
- `g5_vision.py` exposes fixture creation/loading, exact preprocessor invocation, vision-field/conditioning detection, and result normalization without hardcoding unsupported keys.
- `g5_image_authority.py` owns one JAX/model process, hardware gate, one load, Candidate-A verification, memory snapshots, one image-conditioned generation call, and JSON evidence.
- `run_g5_image_authority.sh` owns pre-JAX gates, runtime/dependency capture, attempt budgeting, evidence packaging, and adjudication.

- [ ] Run the failing focused tests.
- [ ] Implement the smallest helper matching observed installed API; keep split vision optional and select native vision if split semantics are not directly supported.
- [ ] Run focused tests to green, then compile Python and syntax-check Bash.
- [ ] Ensure the runner records all required markers, memory/OOM delta, exact path choice, and `G5_TPU_31B_ATTEMPTS_USED` without any G3/G4 rerun.

### Task 4: CPU/static gate and one permitted TPU proof

**Files:**
- Generated only: G5 evidence directory/archive and any narrow source correction required by a direct failure.

- [ ] Run unittest discovery, compileall, Bash syntax checks, JSON parsing, and source-contract checks; record `G5_CPU_PREPARATION=PASS` only on clean exit.
- [ ] Run one primary G5 TPU attempt only if hardware/runtime/dependency gates pass; capture before/after memory and logs.
- [ ] If a direct integration defect occurs, identify the root cause, patch narrowly, rerun all CPU/static checks, and consume at most one corrective TPU attempt on the same healthy allocation.
- [ ] After a PASS, write final adjudication and close G5; after two real failures, stop with `G5_STATUS=OPEN` and budget exhausted.

### Task 5: Final evidence verification and report

**Files:**
- Generated: G5 compact authority package, archive, and `.sha256` sidecar.

- [ ] Verify required evidence filenames, internal and sidecar hashes, JSON markers, fixture hash, source delta classification, and unchanged G3/G4 authority hashes.
- [ ] Run the final verification command set freshly before making completion claims.
- [ ] Report every field required by directive section 26, including exact corrective root cause/fix in at most five lines when applicable.
