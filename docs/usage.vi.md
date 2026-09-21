# Sử dụng

## Khởi động và dừng

```bash
cp .env.example .env
bash scripts/start.sh
python scripts/wait_ready.py
bash scripts/status.sh
bash scripts/stop.sh
```

Nếu cần endpoint public tạm thời, cài `cloudflared` rồi chạy
`bash scripts/run_tunnel.sh`.

## Sinh văn bản

```bash
curl -sS http://127.0.0.1:7860/generate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Giải thích TPU model parallelism.","max_new_tokens":96}'
```

Request dài nên dùng `POST /generate/async` và poll
`GET /result/<job_id>`.

## Image-conditioned generation

Dùng `POST /generate/image` hoặc `/generate/image/async`. JSON nhận
`image_base64`; multipart dùng field `image`.

## Client

Python dùng `clients/python/gemma4_client.py`; Node.js 18+ dùng
`clients/node/gemma4-client.mjs`.
