# Hướng dẫn triển khai

## Yêu cầu

- Kaggle notebook với TPU v5e-8 / v5litepod-8
- attach Gemma 4 31B Keras preset
- Python runtime tương thích với requirements đã pin
- bật Internet khi cần cài dependency hoặc Quick Tunnel

## Cấu hình

Copy `.env.example` thành `.env`. Tối thiểu hãy đặt `API_KEY` và
`RESTART_SECRET` mạnh. Cấu hình mặc định dùng 8 TPU devices, mesh `[1,8]`,
BF16 và Gemma 4 31B preset.

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

Phiên bản TPU đã qualification được ghi trong
[qualification.vi.md](qualification.vi.md).

## Chạy

```bash
bash scripts/start.sh
python scripts/wait_ready.py
curl -sS http://127.0.0.1:7860/health/ready
```

Coordinator tạo một TPU worker sống lâu để chi phí load model và compile đầu
tiên được tái sử dụng cho các request sau.

## Bảo mật

Mặc định API có xác thực. Không publish `.env`, API key, restart secret,
credential model hoặc notebook token.
