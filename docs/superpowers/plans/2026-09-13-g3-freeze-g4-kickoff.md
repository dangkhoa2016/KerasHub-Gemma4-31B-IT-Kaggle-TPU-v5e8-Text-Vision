# Gemma4 G3 Freeze and G4 Kickoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the already-authoritative G3 PASS, publish minimal closeout evidence, resolve the repository-defined G4 scope, and create the authorized G4 kickoff artifact without rerunning G3.

**Architecture:** Read-only authority evidence is the sole source for the G3 result, archive digest, and PASS-producing source hashes. Project artifacts and status documents record that immutable result; G4 is resolved only from canonical project planning documents after the freeze verification passes.

**Tech Stack:** Bash coreutils (`sha256sum`, `tar`, `diff`), JSON validation with Python's standard library, Markdown/JSON project artifacts, and existing CPU/static tests only.

**Spec:** `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-and-g4-kickoff-codex-directive-2026-09-14.md`

## Global Constraints

- Do not run `scripts/run_g3_tpu_authority.sh`, rerun the 31B model, or perform another G3 generation.
- Do not change G3 generation behavior, Candidate-A, R3 sharded checkpoint assignment, or the exact G3 runtime baseline.
- Use only `/kaggle/working/gemma4-31b-vnext-g3-authority-20260913T225039Z/` and its archive/sidecar as G3 authority.
- Do not tag or release; `TAG=false` and `RELEASE=false`.
- Do not copy the large G3 authority archive into the closeout archive.
- Do not begin an expensive TPU workload for G4 unless the canonical repository scope explicitly requires it.

### Task 1: Verify G3 authority and freeze source identity

**Files:**
- Read: `/kaggle/working/gemma4-31b-vnext-g3-authority-20260913T225039Z.tar.gz.sha256`
- Read: `/kaggle/working/gemma4-31b-vnext-g3-authority-20260913T225039Z/07-authority-result.json`
- Read: `/kaggle/working/gemma4-31b-vnext-g3-authority-20260913T225039Z/09-final-adjudication.txt`
- Read: `/kaggle/working/gemma4-31b-vnext-g3-authority-20260913T225039Z/00-source-hashes.txt`
- Verify: current project files listed by `00-source-hashes.txt`

- [x] Verify the authority archive with `sha256sum -c` and require all seven G3 result fields plus `GENERATION_CALL_COUNT=1`.
- [x] Compare every source hash in the authority source-hash file with the current project file using `sha256sum`; stop on any missing file or mismatch.
- [x] Record the exact archive digest and source-hash map for later artifacts; do not reconstruct either from an earlier run.

### Task 2: Create project G3 freeze and closeout documents

**Files:**
- Create: `artifacts/g3/g3-pass-freeze.json`
- Create: `docs/G3-CLOSEOUT.md`

- [x] Write the freeze JSON with the authority path, exact sidecar digest, required G3 fields, and the complete authoritative source-hash map.
- [x] Write the concise closeout with the final Kaggle TPU v5e-8 path, Candidate-A verification, one authority generation call, prompt/max-length facts, and the required compatibility facts.
- [x] Preserve the historical roadmap and source meaning; do not rewrite unrelated sections.

### Task 3: Update canonical project status minimally

**Files:**
- Modify: `DEVELOPMENT-STATUS.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/ROADMAP.vi.md`
- Modify: `CHANGELOG.md`

- [x] Replace stale current-state summaries with `G0=CLOSED`, `G1=CLOSED`, `R3=CLOSED/PASS`, `G2=CLOSED/PASS`, `G3=CLOSED/PASS`, `G3_TEXT_GENERATION=PASS`, `G4_ENTRY_ELIGIBLE=true`, `G4_STARTED=false`, `TAG=false`, and `RELEASE=false`.
- [x] Keep the existing canonical G4 definition as `native-vs-split generation characterization`; do not reinterpret or renumber gates.
- [x] Add only a concise Unreleased G3 closeout entry where the changelog already records project progress.

### Task 4: Run CPU/static freeze verification and publish immutable evidence

**Files:**
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/00-final-authority-reference.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/01-final-authority-sha256-verification.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/02-final-authority-result.json`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/03-pass-source-hashes-authority.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/04-current-source-hashes.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/05-source-hash-verification.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/06-g3-pass-freeze.json`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/07-g3-closeout-doc.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/08-status-diff.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/09-g3-freeze-verification.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/10-final-adjudication.txt`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>/SHA256SUMS`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>.tar.gz`
- Create: `/kaggle/working/gemma4-31b-g3-pass-freeze-closeout-<UTC>.tar.gz.sha256`

- [x] Validate JSON, cross-document status agreement, source hashes, and archive digest without invoking any G3 authority script.
- [x] Run the existing CPU/unit regression suite only after confirming it does not call `scripts/run_g3_tpu_authority.sh`.
- [x] Write the required PASS verification report and final adjudication, then archive only the small closeout directory and verify its own manifest/sidecar.

### Task 5: Resolve canonical G4 scope and kick off G4

**Files:**
- Read: `DEVELOPMENT-STATUS.md`, `docs/ROADMAP.md`, `docs/ROADMAP.vi.md`, `README.md`, `README.vi.md`, and relevant architecture docs
- Create: `artifacts/g4/g4-entry-resolution.md`
- Create: `artifacts/g4/g4-kickoff.json`

- [x] Identify the exact existing G4 definition and cite its canonical source file/section.
- [x] Write `G4_SCOPE_RESOLUTION=PASS` because the canonical documents agree materially.
- [x] Write `G4_STARTED=true` and a concise execution plan limited to the repository-supported native-vs-split generation characterization scope.
- [x] Do not launch TPU work because the current G4 definition does not itself require an immediate expensive workload.

### Task 6: Final independent verification

- [x] Re-read the directive's required final fields and verify each against fresh command output.
- [x] Confirm no prohibited G3 authority command was executed during this turn.
- [x] Report the G3 closeout archive path/digest and exact next G4 action supported by the repository.
