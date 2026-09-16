# Kaggle

Chọn TPU v5e-8 / v5litepod-8 và bật Internet.

Attach Keras Gemma 4 model chứa `gemma4_instruct_31b`. Resolver scan
`/kaggle/input`, không hard-code revision.

Thứ tự:

1. `bash scripts/run_g0_g2.sh`
2. kiểm tra `artifacts/g0-g2/strict-load.json`
3. chỉ sau G2 PASS: `bash scripts/start.sh`
4. `python scripts/wait_ready.py`
5. test text trước, image sau
6. `bash scripts/stop.sh`
