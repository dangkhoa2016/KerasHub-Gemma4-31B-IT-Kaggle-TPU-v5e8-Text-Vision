# Clients

Các client mẫu gọi REST API công khai của server. Mỗi client chỉ dùng thư viện
chuẩn của ngôn ngữ, không cần package bổ sung.

## Python

Yêu cầu Python 3.11 trở lên.

```bash
python clients/python/gemma4_client.py \
  --api-key "$API_KEY" \
  --prompt "Explain TPU model parallelism."
```

Sinh văn bản có điều kiện hình ảnh, truyền đường dẫn ảnh cục bộ:

```bash
python clients/python/gemma4_client.py \
  --api-key "$API_KEY" \
  --prompt "Describe this image." \
  --image /kaggle/input/sample/photo.jpg
```

Client mặc định dùng `POST /generate/async` rồi thăm dò `GET /result/<job_id>`.
Dùng `--url` để trỏ tới endpoint khác, ví dụ sau khi bật Quick Tunnel:

```bash
python clients/python/gemma4_client.py \
  --url https://example.trycloudflare.com \
  --api-key "$API_KEY" \
  --prompt "Explain TPU model parallelism."
```

## Node.js

Yêu cầu Node.js 18 trở lên.

```js
import { generate } from "./clients/node/gemma4-client.mjs";
console.log(await generate("Explain TPU model parallelism."));
```

## Xác thực

Cả hai client đều gửi `Authorization: Bearer <API_KEY>`. Server trả `401` nếu
thiếu hoặc sai khóa. Không ghi khoá thật vào mã nguồn hay vào notebook.

Xem [../docs/api.md](../docs/api.md) hoặc
[../docs/api.vi.md](../docs/api.vi.md) để biết đầy đủ payload và mã lỗi.
