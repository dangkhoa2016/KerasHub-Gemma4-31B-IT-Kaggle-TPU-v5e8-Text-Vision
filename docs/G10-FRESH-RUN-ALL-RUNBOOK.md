# G10 Fresh Kaggle Restart Session → Run All

G10 is the one intentional restart workflow. This document is preparation
only; do not restart the current session during the CPU phase.

## Required sequence

```text
fresh Kaggle Restart Session
→ clone the public canonical repository
→ checkout the exact FINAL_CPU_FROZEN_SHA
→ Run All from the final notebook
→ acquire and verify TPU v5e-8 / eight devices
→ initialize JAX and load gemma4_instruct_31b
→ verify Candidate-A sharding and BF16
→ text acceptance
→ vision acceptance
→ production REST/lifecycle acceptance
→ memory/OOM acceptance
→ compact evidence packaging
```

The notebook must display `git rev-parse HEAD` and fail if it does not equal
the pinned `FINAL_CPU_FROZEN_SHA`. It must not rely on the prior authority
workspace, uncommitted files, local runtime databases, secrets, or caches.

## Fresh-session acceptance

```text
G10_SESSION_FRESH=true
GIT_SHA_EXACT=true
TPU_DEVICE_COUNT=8
MODEL_PRESET=gemma4_instruct_31b
TEXT_ACCEPTANCE=PASS
VISION_ACCEPTANCE=PASS
REST_LIFECYCLE_ACCEPTANCE=PASS
OOM_DELTA=0
EVIDENCE_PACKAGED=true
```

G10 is not complete when only the notebook starts or the model loads. Every
listed acceptance result must be captured from the fresh session.
