# Development status

## Current authoritative state

```text
G0=CLOSED
G1=CLOSED
R3=CLOSED/PASS
G2=CLOSED/PASS
G3=CLOSED/PASS
G3_TEXT_GENERATION=PASS
G3_IMMUTABLE=true
G4=CLOSED/PASS
G4_ENTRY_ELIGIBLE=true
G4_STARTED=false
G4_RECOMMENDED_PATH=SPLIT
G4_IMMUTABLE=true
G5_ENTRY_ELIGIBLE=true
G5=CLOSED/PASS
G5_IMAGE_GENERATION=PASS
G5_PATH=NATIVE_VISION
G5_IMMUTABLE=true
G6_ENTRY_ELIGIBLE=true
G6_STARTED=true
G6=CLOSED/PASS
G6_FINAL_SHARDING=PASS
G6_FINAL_MEMORY_EVIDENCE=PASS
G6_STATUS=CLOSED
G7_ENTRY_ELIGIBLE=true
G7_STARTED=true
G7=CLOSED/PASS
G7_REST_ACCEPTANCE=PASS
G8_ENTRY_ELIGIBLE=true
G8_FINAL_CLOSEOUT=PASS
G8_LIVE_GENERATION_PASS=true
G9_ENTRY_ELIGIBLE=true
G9_STARTED=false
G10_STARTED=false
G11_CPU_PREPARED=true
TAG=false
RELEASE=false
PUBLIC_V1_0_0=false
```

G3 is frozen from the final authority evidence in
`artifacts/g3/g3-pass-freeze.json`. G4 is closed and frozen from the
authoritative characterization evidence in `artifacts/g4/g4-pass-freeze.json`.
G5 is closed and frozen from the image-conditioned generation authority
evidence. G6 is closed from frozen G3/G4/G5 authority evidence without a new
TPU/model execution. G7 and G8 are closed. The final G8 authority records a
successful live generation job, 133 post-fix CPU tests, compileall PASS,
collector corrective PASS, and source provenance PASS; the exact evidence is
preserved under `artifacts/g8/`.

The remaining operational model is `FINAL CPU PREP` followed by one `FINAL TPU
ONE-SHOT`. G9 PRIME/HOT and G10 fresh-session Run All are prepared but not
started. G12 publication is not authorized.

## Historical R1 context

The earlier corrective R1 changed checkpoint-loading semantics so the Gemma 4
task uses KerasHub's native preset loader and routes `model.weights.json` into
`task.backbone`. The final G3 authority source hashes and evidence are the
authoritative state for the current project.
