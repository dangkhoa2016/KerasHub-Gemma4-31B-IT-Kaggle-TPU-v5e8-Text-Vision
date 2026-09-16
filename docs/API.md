# API

Authentication:

```text
Authorization: Bearer <API_KEY>
```

or `X-API-Key: <API_KEY>`.

## Text

`POST /generate/async`

```json
{
  "prompt": "Explain why the sky appears blue.",
  "system": "You are concise.",
  "max_new_tokens": 128
}
```

Poll `GET /result/<job_id>`.

## Image

`POST /generate/image/async`

JSON uses `image_base64`. Multipart uses field `image` plus `prompt`, optional
`system`, and `max_new_tokens`.

## Health

- `GET /health/live`
- `GET /health/ready`
- `GET /info`

## Restart

`POST /restart` requires normal API authentication plus
`X-Restart-Secret`.
