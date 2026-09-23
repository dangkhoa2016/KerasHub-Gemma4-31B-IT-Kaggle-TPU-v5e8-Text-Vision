# Usage

## Start and stop

```bash
cp .env.example .env
bash scripts/start.sh
python scripts/wait_ready.py
bash scripts/status.sh
bash scripts/stop.sh
```

For a temporary public endpoint, install `cloudflared` and run
`bash scripts/run_tunnel.sh`. The tunnel forwards to the local REST server.

## Text generation

```bash
curl -sS http://127.0.0.1:7860/generate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain TPU model parallelism.","max_new_tokens":96}'
```

For long-running requests use `POST /generate/async` and poll
`GET /result/<job_id>`.

## Image-conditioned generation

Use `POST /generate/image` or `/generate/image/async`. JSON requests accept
`image_base64`; multipart requests use an `image` field.

## Clients

```bash
python clients/python/gemma4_client.py --api-key "$API_KEY" \
  --prompt "Explain JAX sharding."
```

Node.js 18+ can use `clients/node/gemma4-client.mjs`.
