# Kaggle Notebook

## Recommended environment

1. Create a Kaggle notebook.
2. Select TPU v5e-8 / v5litepod-8.
3. Enable Internet when required.
4. Attach the Gemma 4 31B Keras model preset.
5. Clone or upload this repository into `/kaggle/working`.
6. Install requirements, configure secrets, and start the server.

The production notebook is stored under `notebooks/`.

## Persistence

Kaggle `/kaggle/working` can be used for project state and JAX compilation
cache within the notebook's persistence model. Treat accelerator allocation
and the running kernel as ephemeral.

## First compile

Gemma 4 31B model load and the first JAX generation compile are expensive.
Keep the worker alive to reuse the executable on later compatible requests.

## Public endpoint

Quick Tunnel is optional. `scripts/run_tunnel.sh` forwards the local server
without requiring a named Cloudflare tunnel or custom domain.
