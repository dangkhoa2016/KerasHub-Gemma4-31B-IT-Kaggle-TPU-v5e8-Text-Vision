# Kaggle

Select TPU v5e-8 / v5litepod-8 and enable Internet.

Attach the Keras Gemma 4 model containing `gemma4_instruct_31b`. The resolver
scans `/kaggle/input` and does not hard-code a revision directory.

Recommended order:

1. `bash scripts/run_g0_g2.sh`
2. inspect `artifacts/g0-g2/strict-load.json`
3. only after G2 PASS: `bash scripts/start.sh`
4. `python scripts/wait_ready.py`
5. test text before image
6. `bash scripts/stop.sh`

The R1 notebook assumes the full source ZIP is uploaded directly by SSH to `/kaggle/working`; no Kaggle Dataset is required for source transport.
