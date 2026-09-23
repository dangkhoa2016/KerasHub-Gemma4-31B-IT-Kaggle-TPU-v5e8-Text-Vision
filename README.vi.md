# KerasHub Gemma 4 31B Instruct trên Kaggle TPU v5e-8

[![CI](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/Backend-JAX-5A45FF.svg)](https://jax.readthedocs.io/)
[![KerasHub](https://img.shields.io/badge/KerasHub-Gemma%204-FF6F00.svg)](https://keras.io/keras_hub/)
[![TPU](https://img.shields.io/badge/Kaggle%20TPU-v5e--8-20A464.svg)](https://www.kaggle.com/)
[![English](https://img.shields.io/badge/lang-English-1f6feb.svg)](README.md)
[![Tiếng Việt](https://img.shields.io/badge/lang-Ti%E1%BA%BFng%20Vi%E1%BB%87t-da251d.svg)](README.vi.md)

Dự án inference Gemma 4 31B Instruct theo hướng production trên Kaggle TPU
v5e-8, sử dụng KerasHub, Keras, JAX, ModelParallel sharding, sinh văn bản,
image-to-text, REST API có xác thực, job bất đồng bộ và JAX compilation cache.

> Đây là dự án kỹ thuật độc lập xây dựng quanh model Gemma upstream và runtime
> KerasHub. Đây không phải bản phát hành chính thức của Google, Kaggle, Keras
> hay OpenAI.

## Điểm chính

- Gemma 4 31B Instruct qua `keras_hub.models.Gemma4CausalLM`
- một logical model được shard trên 8 TPU devices
- BF16 và strict checkpoint loading
- text generation và image-conditioned text generation
- JAX decode loop ổn định để tái sử dụng executable ở hot request
- REST API sync + async
- request ID, API key, restart secret, queue limit và TTL
- client Python và Node.js
- script hỗ trợ Kaggle TPU và production notebook
- tài liệu song ngữ Anh / Việt

## Runtime đã qualification

```text
Model             gemma4_instruct_31b
Backend           JAX
Keras             3.15.1
KerasHub          0.31.1
JAX / JAXLIB      0.10.2 / 0.10.2
libtpu            0.0.17
Accelerator       Kaggle TPU v5e-8
TPU devices       8
Mesh              [1, 8]
Axes              [batch, model]
Dtype             bfloat16
```

Qualification cuối trên production source ghi nhận request đầu mất khoảng
541,4 giây do compile, còn request hot giống hệt mất khoảng 5,24 giây và không
phát sinh compile event mới. Đây là số đo của request/runtime đã qualification,
không phải cam kết latency cho mọi prompt hoặc request đa phương thức.

## Khởi chạy nhanh

```bash
git clone https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision.git
cd KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision

cp .env.example .env
# Điền API_KEY và RESTART_SECRET vào .env.

bash scripts/start.sh
python scripts/wait_ready.py
```

Trên Kaggle cần attach Gemma 4 31B Keras preset và bật TPU v5e-8 /
v5litepod-8. Hãy đọc
[docs/kaggle-notebook.vi.md](docs/kaggle-notebook.vi.md) trước lần chạy TPU đầu.

## Ví dụ API

```bash
curl -sS http://127.0.0.1:7860/generate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Giải thích TPU model parallelism trong ba câu.",
    "max_new_tokens": 96
  }'
```

Image-conditioned generation dùng `/generate/image` hoặc
`/generate/image/async`.

## Tài liệu

| Chủ đề | English | Tiếng Việt |
|---|---|---|
| Mục lục | [docs/README.md](docs/README.md) | [docs/README.vi.md](docs/README.vi.md) |
| Sử dụng | [usage](docs/usage.md) | [sử dụng](docs/usage.vi.md) |
| Cài đặt | [guide](docs/guide.md) | [hướng dẫn](docs/guide.vi.md) |
| REST API | [api](docs/api.md) | [api](docs/api.vi.md) |
| Kiến trúc | [architecture](docs/architecture.md) | [kiến trúc](docs/architecture.vi.md) |
| Kaggle notebook | [kaggle notebook](docs/kaggle-notebook.md) | [kaggle notebook](docs/kaggle-notebook.vi.md) |
| Giới hạn | [limitations](docs/limitations.md) | [giới hạn](docs/limitations.vi.md) |
| Kiến thức kỹ thuật | [knowledge](docs/knowledge.md) | [kiến thức](docs/knowledge.vi.md) |
| Xử lý sự cố | [troubleshooting](docs/troubleshooting.md) | [xử lý sự cố](docs/troubleshooting.vi.md) |
| Qualification | [qualification](docs/qualification.md) | [qualification](docs/qualification.vi.md) |

## Cấu trúc repository

```text
.github/      workflow, template và repository policy
clients/      client Python và Node.js
docs/         tài liệu public song ngữ
evidence/     tóm tắt qualification public
notebooks/    Kaggle production notebook
scripts/      script vận hành và verification
src/          production server và TPU runtime
tests/        unit/contract test theo hành vi
```

## License

MIT. Xem [LICENSE](LICENSE) và [NOTICE](NOTICE).
