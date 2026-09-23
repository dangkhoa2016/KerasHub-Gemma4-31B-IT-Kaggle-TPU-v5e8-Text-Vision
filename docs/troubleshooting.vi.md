# Xử lý sự cố

## TPU device count không phải 8

Kiểm tra accelerator là TPU v5e-8 / v5litepod-8 và không có process JAX cũ đang
giữ device handle. Không chạy nhiều JAX TPU client độc lập trên cùng allocation.

## Worker chưa ready

Xem `logs/server.stdout.log`, `/health/ready` và worker load timeout. Load model
có thể mất nhiều phút.

## Request đầu rất chậm

Request đầu với shape mới có thể phải compile executable lớn. Giữ worker sống và
thử lại cùng shape trước khi kết luận hot latency chậm.

## Memory tăng mạnh khi load

Không load nhiều model 31B song song. Kiểm tra cgroup memory và checkpoint page
cache.

## 401 Unauthorized

Gửi API key qua `Authorization: Bearer ...` hoặc `X-API-Key`. Restart còn cần
`X-Restart-Secret`.
