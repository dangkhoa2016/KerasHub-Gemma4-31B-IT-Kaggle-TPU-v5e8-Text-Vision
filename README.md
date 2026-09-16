# KerasHub Gemma 4 31B Instruct on Kaggle TPU v5e-8 — Text + Vision

Standalone successor project derived from the architecture lessons of the
completed TranslateGemma 27B project. This is a **new independent source tree**.

> **G8 final closeout:** the async/lifecycle CPU qualification and final live generation authority are closed. Frozen evidence is recorded in `artifacts/g8/`, including the post-fix and formal closeout packages.

## Current status

```text
G0 standalone source tree                    CLOSED
G1 TPU/model preflight                       CLOSED
R3 sharded checkpoint assignment              CLOSED/PASS
G2 strict checkpoint load                    CLOSED/PASS
G3 text generation                           CLOSED/PASS
G4 generation architecture characterization CLOSED/PASS (SPLIT)
G5 image generation                          CLOSED/PASS (NATIVE_VISION)
G6 final sharding + memory evidence          CLOSED/PASS
G7 REST server                               CLOSED/PASS (CPU REST contract)
G8 async/cold compile/lifecycle              CLOSED/PASS
G8 final live generation                     PASS
G9 PRIME/HOT                                 ELIGIBLE / NOT STARTED
G10 fresh Restart Session -> Run All         NOT STARTED
G11 history/source hardening                 CPU PREPARATION IN PROGRESS
G12 public v1.0.0                            NOT RELEASED
```

The remaining operational work is intentionally compressed to two phases:

```text
STEP 1  FINAL CPU PREP       source/history/release preparation
STEP 2  FINAL TPU ONE-SHOT   G9 PRIME/HOT and G10 fresh-session validation
```

The final G8 live authority recorded `G9_ENTRY_ELIGIBLE=true` and
`G9_STARTED=false`. PRIME/HOT, fresh-session validation, tagging, and release
publication remain deliberately open.

## Target

```text
preset          gemma4_instruct_31b
model class     keras_hub.models.Gemma4CausalLM
backend         JAX
hardware        Kaggle TPU v5e-8 / v5litepod-8
TPU devices     8
logical model   1
mesh            [1,8]
axes            [batch, model]
dtype           bfloat16
load            strict / skip_mismatch=False
text            yes
image/vision    yes
audio           no for this 31B target
```

The old Gemma3/TranslateGemma split-prefill/decode engine is not copied blindly.
Gemma 4 has different cache/attention/vision semantics. G3/G5 initially use
KerasHub native generation only for characterization. If real TPU evidence shows
unsafe compile/host-memory behavior, G4 will replace it with a Gemma4-native
split engine.

## First run

```bash
cp .env.example .env
bash scripts/run_g0_g2.sh
```

Do not start REST acceptance before `artifacts/g0-g2/strict-load.json` passes.

## REST endpoints

```text
GET  /
GET  /health/live
GET  /health/ready
GET  /info
POST /generate
POST /generate/async
POST /generate/image
POST /generate/image/async
GET  /result/<job_id>
POST /restart
```

See `docs/KAGGLE.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, and
`docs/ROADMAP.md`.
