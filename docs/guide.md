# Deployment Guide

## Requirements

- Kaggle notebook with TPU v5e-8 / v5litepod-8
- attached Gemma 4 31B Keras preset
- Python runtime compatible with the pinned requirements
- Internet enabled when dependencies or Quick Tunnel are needed

## Configure

Copy `.env.example` to `.env`. At minimum set strong values for
`API_KEY` and `RESTART_SECRET`. The default target expects eight TPU devices,
a `[1,8]` ModelParallel mesh, BF16, and the Gemma 4 31B preset.

## Install

```bash
python -m pip install -r requirements.txt
```

TPU-specific versions used by the qualified runtime are documented in
[qualification.md](qualification.md).

## Run

```bash
bash scripts/start.sh
python scripts/wait_ready.py
curl -sS http://127.0.0.1:7860/health/ready
```

The coordinator starts a long-lived spawned TPU worker. Model loading and the
first compile are intentionally amortized across later requests.

## Security

Authentication is enabled by default. Never publish `.env`, generated API
keys, restart secrets, model credentials, or notebook tokens.
