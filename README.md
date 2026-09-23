# KerasHub Gemma 4 31B Instruct on Kaggle TPU v5e-8 — Text + Vision

Standalone successor project derived from the architecture lessons of the
completed TranslateGemma 27B project. This is a **new independent source tree**.

> **Current authority:** G0-G10 are closed, the post-G10 generation-loop
> corrective is TPU-qualified, and the corrective source has passed public CI.
> Public v1.0.0 publication remains intentionally open until the final
> publication review is complete.

## Current status

```text
G0 standalone source tree                    CLOSED
G1 TPU/model preflight                       CLOSED
R3 sharded checkpoint assignment             CLOSED/PASS
G2 strict checkpoint load                    CLOSED/PASS
G3 text generation                           CLOSED/PASS
G4 generation architecture characterization  CLOSED/PASS
G5 image generation                          CLOSED/PASS (NATIVE_VISION)
G6 final sharding + memory evidence           CLOSED/PASS
G7 REST server                               CLOSED/PASS
G8 async/cold compile/lifecycle              CLOSED/PASS
G9 PRIME/HOT                                 CLOSED/PASS
G10 fresh Restart Session -> Run All         CLOSED/PASS
Post-G10 generation-loop corrective          CLOSED/PASS
G11 history/source hardening                 CLOSED/PASS
G12 public v1.0.0                            NOT RELEASED
```

The remaining work is publication-only:

```text
STEP 1  FINAL PUBLICATION REVIEW   verify docs, source authority, CI and release metadata
STEP 2  PUBLIC v1.0.0             create tag/release only after the review closes
```

The production corrective keeps `run_eagerly=True` and uses a stable JAX
`lax.while_loop` callable identity inside `StableGemma4GreedySampler`.
In the production-source TPU qualification, the first identical request paid
the compile cost, while the second identical hot request reused the executable:

```text
warm request              541.409 s
hot identical request       5.239 s
warm-to-hot speedup        103.34x
hot compile attempts         0
same output                 true
```

These are qualification results for the tested request and runtime, not a
general latency guarantee for every prompt length or multimodal request.

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
Gemma 4 keeps KerasHub-native cache/prefill semantics, while the generation loop
uses a stable greedy sampler so hot requests can reuse the compiled JAX
while-loop executable instead of rebuilding it per request.

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
