# REST API

Authentication uses either:

```text
Authorization: Bearer <API_KEY>
X-API-Key: <API_KEY>
```

## Health and metadata

- `GET /`
- `GET /health/live`
- `GET /health/ready`
- `GET /info`

## Text

### `POST /generate`

```json
{
  "prompt": "Explain model parallelism.",
  "system": "Answer concisely.",
  "max_new_tokens": 128
}
```

### `POST /generate/async`

Returns HTTP 202 with `job_id` and `result_url`.

## Vision

- `POST /generate/image`
- `POST /generate/image/async`

JSON uses `image_base64`; multipart uses field `image` plus optional
`prompt`, `system`, and `max_new_tokens`.

## Results

`GET /result/<job_id>` returns 202 while queued/processing, 200 on success,
404 when missing/expired, and 500 for a failed completed job.

## Restart

`POST /restart` requires normal API authentication plus `X-Restart-Secret`.
The restart operation is serialized and returns HTTP 202 when accepted.

Every response includes `X-Request-ID`.
