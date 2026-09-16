# Roadmap

## Current gate state

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
G4_RECOMMENDED_PATH=SPLIT
G4_STARTED=false
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

## Final CPU / TPU boundary

The remaining project is intentionally limited to two operational phases:

```text
STEP 1  FINAL CPU PREP       source/history/release preparation
STEP 2  FINAL TPU ONE-SHOT   G9 PRIME/HOT and G10 fresh-session Run All
```

G0-G8 are closed. G9 and G10 are not complete and must be run only by the
final TPU one-shot using the exact frozen Git SHA. G11 preparation is allowed
in the CPU phase; G12 publication is not authorized until G10 passes.

```text
G0  standalone source identity
G1  TPU v5e-8 preflight + model discovery
G2  Gemma4 31B ModelParallel strict-load proof
G3  shortest possible text generation proof
G4  native-vs-split generation characterization
G5  image-conditioned generation proof
G6  final sharding + memory evidence
G7  REST server acceptance
G8  async/cold compile + restart/lifecycle acceptance
G9  PRIME/HOT acceptance
G10 fresh Kaggle Restart Session -> Run All
G11 history/source hardening
G12 public v1.0.0
```

Do not publish v1.0.0 before G10 closes.
