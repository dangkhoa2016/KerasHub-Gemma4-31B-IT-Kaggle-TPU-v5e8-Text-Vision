# KerasHub Gemma 4 31B Instruct on Kaggle TPU v5e-8

[![CI](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision)](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/Backend-JAX-5A45FF.svg)](https://jax.readthedocs.io/)
[![KerasHub](https://img.shields.io/badge/KerasHub-Gemma%204-FF6F00.svg)](https://keras.io/keras_hub/)
[![TPU](https://img.shields.io/badge/Kaggle%20TPU-v5e--8-20A464.svg)](https://www.kaggle.com/)
[![English](https://img.shields.io/badge/lang-English-1f6feb.svg)](README.md)
[![Tiếng Việt](https://img.shields.io/badge/lang-Ti%E1%BA%BFng%20Vi%E1%BB%87t-da251d.svg)](README.vi.md)

A reproducible, production-oriented inference stack for **Gemma 4 31B Instruct** on **Kaggle TPU v5e-8**, implemented with **KerasHub + Keras + JAX** and designed to demonstrate more than a one-cell inference example.

The project runs one logical 31B multimodal model across **8 TPU devices** with Keras ModelParallel sharding, BF16 execution, text generation, image-conditioned text generation, authenticated REST endpoints, synchronous and asynchronous jobs, client implementations, operational scripts, reproducibility guidance, and preserved public execution evidence.

> This is an independent engineering project built around the upstream Gemma/Keras ecosystem. It is not an official Google, Kaggle, Keras, or OpenAI release.

## Why this project exists

Large-model demos often stop at “the model loaded and produced one answer.” This repository focuses on the harder engineering questions around a repeatable TPU workflow:

- Can a **31B multimodal model** be loaded through the intended KerasHub path on Kaggle TPU?
- Can one logical model be distributed across **all 8 TPU devices** instead of treated as eight independent replicas?
- Can the runtime serve both **text and vision-conditioned requests** through the same production-oriented stack?
- Can compatible repeated requests reuse a stable JAX decode execution path?
- Can the project expose practical **sync/async REST APIs**, authentication, restart protection, request IDs, queue limits, and clients?
- Can a reviewer independently inspect the source, import the notebook, attach the exact model variation, run it, and compare results against preserved public evidence?

The `v1.0.0` release answers those questions with source code, documentation, a public Kaggle showcase, qualification evidence, and downloadable release artifacts.

## Release snapshot at a glance

| Item | Qualified / published result |
|---|---|
| Model source | `keras/gemma4/Keras/gemma4_instruct_31b/2` |
| Accelerator | Kaggle TPU v5e-8 |
| TPU devices | **8** |
| Logical mesh | **[1, 8]** |
| Dtype | **BF16** |
| Public text requests | **6** |
| Public vision requests | **5** |
| Completed requests | **11/11** |
| Notebook code cells executed | **24/24** |
| Output-bearing code cells | **23** |
| Cell errors | **0** |
| Final marker | **`SHOWCASE=PASS`** |
| Mean text inference | **16.244 s** |
| Mean vision inference | **15.175 s** |
| Public identical-text hot speedup | **3.728×** |
| Dedicated compatible-request speedup | **6.7015×** |
| Dedicated hot compile events | **0** |
| Model load count | **1** |

The latency figures above are evidence from the completed qualification/showcase runs. They are **not** an SLA and should not be interpreted as universal latency guarantees for arbitrary prompts, images, token counts, or future Kaggle sessions.

## Architecture highlights

### 1. One logical model across 8 TPU devices

The project uses Keras distribution APIs and ModelParallel sharding so the Gemma 4 31B model is distributed over a logical mesh of:

```text
[batch=1, model=8]
```

This is model parallelism for one model instance, not eight independent model replicas.

### 2. KerasHub preset loading

The release is built around the upstream Keras model variation:

```text
keras/gemma4/Keras/gemma4_instruct_31b/2
```

and loads Gemma through `keras_hub.models.Gemma4CausalLM` with strict checkpoint handling.

### 3. Text + vision in one runtime

The service supports:

- text-only generation;
- image-conditioned text generation;
- synchronous endpoints for immediate requests;
- asynchronous jobs for queued work;
- Python and Node.js clients using the same API contract.

### 4. Production-oriented service behavior

The repository includes more than model loading code:

- API-key authentication;
- request IDs;
- protected worker restart using a restart secret;
- queue limits and TTL behavior;
- health/readiness handling;
- synchronous and asynchronous generation routes;
- operational helpers for Kaggle sessions;
- behavior-focused unit and contract tests.

### 5. Stable hot-request execution path

A dedicated compatible-request qualification recorded:

```text
Warm request             38.651967787 s
Identical hot request     5.767615575 s
Speedup                   6.701550629×
Model load count          1
Same model object         True
Same output               True
Hot compile attempts      0
Hot compile events        0
Hot effective compile     0
OOM                       0
```

The important result is not merely the lower hot latency: the same logical model remained loaded and the qualified repeated request reused the stable decode execution path without observing a new compile event.

## Qualified runtime

```text
Model source       keras/gemma4/Keras/gemma4_instruct_31b/2
Backend            JAX
Keras              3.15.1
KerasHub           0.32.0
JAX                0.10.2
JAXLIB             0.10.2
libtpu             0.0.49
Accelerator        Kaggle TPU v5e-8
TPU devices        8
Mesh               [1, 8]
Axes               [batch, model]
Dtype              bfloat16
```

## Public Kaggle production showcase

Canonical executed notebook:

**https://www.kaggle.com/code/dangkhoa2016/gemma-4-31b-kaggle-tpu-v5e-8-production-showcase**

The public **Version 1** notebook preserves an executed TPU v5e-8 showcase with:

- **6 text requests**;
- **5 vision requests**;
- **11 completed requests**;
- **24/24 executed code cells**;
- **23 output-bearing code cells**;
- **0 cell errors**;
- `SHOWCASE=PASS`;
- `COMPLETED_REQUESTS=11`.

The vision workload combines deterministic visual checks with a photorealistic synthetic portrait to exercise both structured visual reasoning and natural-scene/attribute grounding.

Recorded run-specific showcase timings:

```text
Server startup                 1920.829 s
Mean text inference              16.244 s
Mean vision inference            15.175 s
Identical-text hot speedup        3.728×
```

These measurements are intentionally preserved as evidence from that run rather than presented as a performance guarantee.

## Reproducibility: source template vs executed evidence

The project deliberately keeps two complementary notebook surfaces.

### GitHub source notebook

[`notebooks/kaggle-tpu-v5e8-text-vision-production-showcase.ipynb`](notebooks/kaggle-tpu-v5e8-text-vision-production-showcase.ipynb)

Role:

- clean, inspectable reproduction template;
- suitable for importing into Kaggle;
- documents exact setup and expected PASS markers;
- pins `RUNTIME_REF = 'v1.0.0'` so future runs do not silently execute an arbitrary future `main` branch.

### Kaggle executed notebook

Role:

- public execution evidence;
- preserves real text and vision outputs;
- preserves completed-run timing/runtime metadata;
- allows reviewers to compare their own run with an executed reference.

These notebooks are not expected to be byte-identical. Kaggle reserializes notebook JSON, and independent reruns can also differ in wall-clock timings and generated wording. Review should focus on the model variation, runtime tag, workflow, code logic, PASS markers, request classes, and semantic behavior.

## Reproduce on Kaggle

A reviewer can follow this path:

1. Open the GitHub source notebook.
2. Import it into Kaggle using **Copy & Edit**.
3. Enable **TPU v5e-8**.
4. Enable Internet access.
5. Attach exactly:
   `keras/gemma4/Keras/gemma4_instruct_31b/2`
6. Verify the model input mount.
7. Start from a clean session if accelerator/model input settings changed.
8. Run all cells.
9. Verify the expected markers and real outputs.
10. Compare behavior with the public executed Version 1 notebook and release artifacts.

Expected model mount:

```text
/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2
```

Expected markers include:

```text
TPU_CONFIG_IMPORTED=PASS
TPU_PREFLIGHT=PASS
SERVER_READY=PASS
SHOWCASE=PASS
COMPLETED_REQUESTS=11
```

## Quick start

```bash
git clone https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision.git
cd KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision

cp .env.example .env
# Add API_KEY and RESTART_SECRET to .env.

bash scripts/start.sh
python scripts/wait_ready.py
```

For Kaggle TPU execution, follow [docs/kaggle-notebook.md](docs/kaggle-notebook.md) rather than treating the shell quick-start alone as sufficient setup.

## API examples

### Text generation

```bash
curl -sS http://127.0.0.1:7860/generate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain TPU model parallelism in three sentences.",
    "max_new_tokens": 96
  }'
```

### Vision-conditioned generation

Image-conditioned generation is exposed through:

```text
/generate/image
/generate/image/async
```

The asynchronous API is intended for workflows that should not keep one HTTP request open while generation is queued or running.

## Release evidence

The `v1.0.0` GitHub Release includes reviewer-facing artifacts for the executed public showcase:

- `gemma4-v1.0.0-public-showcase-evidence.zip`
- `kaggle-tpu-v5e8-production-executed-public.html`
- `kaggle-tpu-v5e8-production-executed-public.ipynb`
- `showcase-results-public.json`

Canonical release provenance is anchored by the annotated `v1.0.0` tag and the GitHub Release. The exact release commit, source tree, artifact checksums, and reviewer-facing evidence are published there so they can be verified without creating a self-referential commit hash inside this README.

The release evidence is intended to make the project reviewable without asking readers to trust screenshots or undocumented local runs.

## Documentation

| Topic | English | Tiếng Việt |
|---|---|---|
| Documentation index | [docs/README.md](docs/README.md) | [docs/README.vi.md](docs/README.vi.md) |
| Usage | [usage](docs/usage.md) | [sử dụng](docs/usage.vi.md) |
| Setup guide | [guide](docs/guide.md) | [hướng dẫn](docs/guide.vi.md) |
| REST API | [api](docs/api.md) | [api](docs/api.vi.md) |
| Architecture | [architecture](docs/architecture.md) | [kiến trúc](docs/architecture.vi.md) |
| Kaggle notebook | [kaggle notebook](docs/kaggle-notebook.md) | [kaggle notebook](docs/kaggle-notebook.vi.md) |
| Limitations | [limitations](docs/limitations.md) | [giới hạn](docs/limitations.vi.md) |
| Technical knowledge | [knowledge](docs/knowledge.md) | [kiến thức](docs/knowledge.vi.md) |
| Troubleshooting | [troubleshooting](docs/troubleshooting.md) | [xử lý sự cố](docs/troubleshooting.vi.md) |
| Qualification evidence | [qualification](docs/qualification.md) | [qualification](docs/qualification.vi.md) |

## Repository layout

```text
.github/      GitHub workflows, templates and repository policy
clients/      Python and Node.js clients
docs/         bilingual public documentation
evidence/     compact public qualification summary
notebooks/    Kaggle production notebook
scripts/      operational and verification helpers
src/          production server and TPU runtime
tests/        behavior-focused unit and contract tests
```

## Scope and limitations

This project demonstrates a validated engineering path for the recorded runtime and workload. It does not claim:

- universal latency across prompts or future Kaggle sessions;
- an SLA;
- deterministic token-for-token outputs for every rerun;
- support for arbitrary Gemma variants without requalification;
- endorsement by Google, Kaggle, Keras, or OpenAI.

For detailed operational constraints, see [docs/limitations.md](docs/limitations.md).

## Release

Current production release: **[v1.0.0](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/releases/tag/v1.0.0)**.

The release is the stable baseline used by the reproduction notebook and public evidence workflow.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
