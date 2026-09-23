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
G5=CLOSED/PASS
G5_IMAGE_GENERATION=PASS
G5_PATH=NATIVE_VISION
G6=CLOSED/PASS
G6_FINAL_SHARDING=PASS
G6_FINAL_MEMORY_EVIDENCE=PASS
G7=CLOSED/PASS
G7_REST_ACCEPTANCE=PASS
G8=CLOSED/PASS
G8_FINAL_CLOSEOUT=PASS
G8_LIVE_GENERATION_PASS=true
G9=CLOSED/PASS
G10=CLOSED/PASS
POST_G10_GENERATION_LOOP_CORRECTIVE=CLOSED/PASS
G11=CLOSED/PASS
TAG=false
RELEASE=false
PUBLIC_V1_0_0=false
```

## Publication boundary

All runtime acceptance gates through G10 are closed. The post-G10 generation
latency corrective is also closed and TPU-qualified. The remaining work is
limited to final publication review and G12 publication.

```text
G0  standalone source identity
G1  TPU v5e-8 preflight + model discovery
G2  Gemma4 31B ModelParallel strict-load proof
G3  shortest possible text generation proof
G4  generation architecture characterization
G5  image-conditioned generation proof
G6  final sharding + memory evidence
G7  REST server acceptance
G8  async/cold compile + restart/lifecycle acceptance
G9  PRIME/HOT acceptance
G10 fresh Kaggle Restart Session -> Run All
G11 history/source hardening and post-corrective authority
G12 public v1.0.0
```

## Post-G10 corrective authority

The production engine keeps `run_eagerly=True` and uses
`StableGemma4GreedySampler` to keep the JAX `lax.while_loop` callable
identity stable across requests.

The production-source qualification recorded:

```text
model_load_count=1
same_model_object=true
same_output=true
warm_seconds=541.409262515
hot_seconds=5.239080924
speedup=103.340503873
hot_compile_attempt_count=0
hot_compile_event_count=0
oom_counter_increase=0
watchdog_triggered=false
```

No additional TPU rerun is required for documentation-only publication
preparation as long as production source remains unchanged.

Do not create the public v1.0.0 tag/release until the final publication review
confirms documentation, CI, source authority, and release metadata are aligned.
