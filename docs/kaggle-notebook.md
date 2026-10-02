# Kaggle Production Showcase

Official public notebook: [https://www.kaggle.com/code/dangkhoa2016/gemma-4-31b-kaggle-tpu-v5e-8-production-showcase](https://www.kaggle.com/code/dangkhoa2016/gemma-4-31b-kaggle-tpu-v5e-8-production-showcase)

## Recommended environment

1. Create a Kaggle notebook.
2. Select TPU v5e-8 / v5litepod-8.
3. Enable Internet when required.
4. Attach the Gemma 4 31B Keras model preset.
5. Open `notebooks/kaggle-tpu-v5e8-text-vision-production-showcase.ipynb` and choose **Run All**.

## What the notebook demonstrates

The notebook is intentionally broader than a smoke test. A full showcase run covers:

- TPU preflight and live runtime metadata;
- six text/chat-style requests, including English, Vietnamese, code generation,
  structured instruction following and an identical hot repeat;
- a richer deterministic visual diagnostic card generated with Pillow;
- three diagnostic-card image-conditioned requests covering scene understanding,
  spatial grounding/counting and visible-text/table reading;
- two photorealistic portrait image-conditioned requests using a synthetic
  Mage-Flow-Turbo-generated portrait for natural-scene and attribute grounding;
- an optional user-provided image through `SHOWCASE_IMAGE`;
- per-request wall time, inference time, generation bucket information and
  compile-cache evidence;
- a machine-readable summary at `/kaggle/working/gemma4-showcase-results.json`.

Gemma 4 in this project consumes images and generates text. It is **not** a
text-to-image generator.

## Performance interpretation

Model loading and the first compile are expensive. Keep the worker alive so
compatible requests can reuse compiled executables. The prior qualification
recorded about 541.4 seconds for the first compatible request and about 5.24
seconds for an identical hot request. The showcase reports fresh measurements
for its own prompts; those qualification values are not treated as an SLA.

## Persistence

Kaggle `/kaggle/working` can hold project state, the generated visual test card,
the showcase result JSON and JAX compilation cache within Kaggle's persistence
model. Treat the active accelerator and kernel as ephemeral.

## Optional user image

To add a real image after the deterministic vision demo, set `SHOWCASE_IMAGE` to
a `.jpg`, `.jpeg`, `.png`, or `.webp` file under `/kaggle/input`.

## Public endpoint

Quick Tunnel remains optional. `scripts/run_tunnel.sh` forwards the local server
without requiring a named Cloudflare tunnel or custom domain.