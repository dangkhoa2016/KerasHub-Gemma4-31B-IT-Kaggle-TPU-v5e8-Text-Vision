# Roadmap

## Trạng thái gate hiện tại

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

## Ranh giới FINAL CPU / TPU

Phần còn lại chỉ gồm hai phase vận hành:

```text
STEP 1  FINAL CPU PREP       chuẩn bị source/history/release
STEP 2  FINAL TPU ONE-SHOT   G9 PRIME/HOT và G10 Run All session mới
```

G0-G8 đã đóng. G9 và G10 chưa hoàn tất và chỉ được chạy bởi final TPU
one-shot với Git SHA đã freeze. G11 được phép chuẩn bị trong CPU phase; G12
chưa được publish trước khi G10 PASS.

```text
G0  standalone source identity
G1  TPU v5e-8 preflight + model discovery
G2  Gemma4 31B ModelParallel strict-load proof
G3  text generation proof tối thiểu
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

Không public v1.0.0 trước khi G10 đóng.
