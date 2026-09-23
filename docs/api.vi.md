# REST API

Xác thực bằng một trong hai cách:

```text
Authorization: Bearer <API_KEY>
X-API-Key: <API_KEY>
```

## Health và metadata

- `GET /`
- `GET /health/live`
- `GET /health/ready`
- `GET /info`

## Text

`POST /generate` chạy đồng bộ. `POST /generate/async` trả HTTP 202 cùng
`job_id` và `result_url`.

## Vision

- `POST /generate/image`
- `POST /generate/image/async`

JSON dùng `image_base64`; multipart dùng field `image`.

## Kết quả

`GET /result/<job_id>` trả 202 khi còn xử lý, 200 khi thành công, 404 nếu job
không tồn tại/hết hạn và 500 nếu job hoàn tất với lỗi.

## Restart

`POST /restart` cần API auth và `X-Restart-Secret`. Mọi response có
`X-Request-ID`.
