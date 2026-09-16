# API

Authentication dùng `Authorization: Bearer <API_KEY>` hoặc
`X-API-Key: <API_KEY>`.

Text async: `POST /generate/async`.

Image async: `POST /generate/image/async`.

Poll kết quả bằng `GET /result/<job_id>`.

Health: `/health/live`, `/health/ready`, `/info`.

`POST /restart` cần thêm header `X-Restart-Secret`.
