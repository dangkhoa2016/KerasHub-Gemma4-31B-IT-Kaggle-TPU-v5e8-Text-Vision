# KerasHub Gemma 4 31B Instruct trên Kaggle TPU v5e-8

[![CI](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision)](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/Backend-JAX-5A45FF.svg)](https://jax.readthedocs.io/)
[![KerasHub](https://img.shields.io/badge/KerasHub-Gemma%204-FF6F00.svg)](https://keras.io/keras_hub/)
[![TPU](https://img.shields.io/badge/Kaggle%20TPU-v5e--8-20A464.svg)](https://www.kaggle.com/)
[![English](https://img.shields.io/badge/lang-English-1f6feb.svg)](README.md)
[![Tiếng Việt](https://img.shields.io/badge/lang-Ti%E1%BA%BFng%20Vi%E1%BB%87t-da251d.svg)](README.vi.md)

Một inference stack **Gemma 4 31B Instruct** theo hướng production và có thể tái lập trên **Kaggle TPU v5e-8**, được xây dựng bằng **KerasHub + Keras + JAX** và hướng tới nhiều hơn một demo “load model rồi sinh một câu trả lời”.

Dự án chạy một logical multimodal model 31B trên **8 TPU devices** bằng Keras ModelParallel sharding, BF16, text generation, image-conditioned text generation, REST API có xác thực, job sync/async, client implementations, operational scripts, hướng dẫn reproducibility và public execution evidence được lưu lại để reviewer có thể kiểm chứng.

> Đây là một dự án kỹ thuật độc lập xây dựng quanh hệ sinh thái Gemma/Keras upstream. Đây không phải release chính thức của Google, Kaggle, Keras hay OpenAI.

## Vì sao dự án này tồn tại

Nhiều demo model lớn dừng lại ở mức “model load được và trả lời được một prompt”. Repository này tập trung vào những câu hỏi khó hơn ở góc độ engineering và reproducibility:

- Một **multimodal model 31B** có thể load đúng qua KerasHub trên Kaggle TPU hay không?
- Một logical model có thể được phân phối trên **đủ 8 TPU devices** thay vì biến thành 8 replica độc lập hay không?
- Cùng một runtime có thể phục vụ cả **text** và **vision-conditioned requests** hay không?
- Compatible repeated requests có thể tái sử dụng stable JAX decode execution path hay không?
- Có thể bổ sung **sync/async REST APIs**, authentication, restart protection, request IDs, queue limits và clients theo hướng production hay không?
- Reviewer có thể tự inspect source, import notebook, attach đúng model variation, chạy lại và so sánh với public executed evidence hay không?

Release `v1.0.0` trả lời các câu hỏi đó bằng source code, tài liệu song ngữ, public Kaggle showcase, qualification evidence và downloadable release artifacts.

## Toàn cảnh release

| Hạng mục | Kết quả đã qualification / publish |
|---|---|
| Model source | `keras/gemma4/Keras/gemma4_instruct_31b/2` |
| Accelerator | Kaggle TPU v5e-8 |
| TPU devices | **8** |
| Logical mesh | **[1, 8]** |
| Dtype | **BF16** |
| Public text requests | **6** |
| Public vision requests | **5** |
| Completed requests | **11/11** |
| Notebook code cells executed | **24/24** |
| Output-bearing code cells | **23** |
| Cell errors | **0** |
| Final marker | **`SHOWCASE=PASS`** |
| Mean text inference | **16.244 s** |
| Mean vision inference | **15.175 s** |
| Public identical-text hot speedup | **3.728×** |
| Dedicated compatible-request speedup | **6.7015×** |
| Dedicated hot compile events | **0** |
| Model load count | **1** |

Các con số latency ở trên là evidence của những run đã hoàn tất. Chúng **không phải SLA** và không nên được hiểu là cam kết latency cố định cho mọi prompt, image, token count hoặc mọi Kaggle session trong tương lai.

## Điểm nổi bật về kiến trúc

### 1. Một logical model trên 8 TPU devices

Dự án sử dụng Keras distribution APIs và ModelParallel sharding để phân phối Gemma 4 31B trên logical mesh:

```text
[batch=1, model=8]
```

Đây là model parallelism cho một model instance, không phải tám model replica độc lập.

### 2. Load đúng KerasHub preset

Release sử dụng upstream Keras model variation:

```text
keras/gemma4/Keras/gemma4_instruct_31b/2
```

và load Gemma qua `keras_hub.models.Gemma4CausalLM` với strict checkpoint handling.

### 3. Text + vision trong cùng runtime

Service hỗ trợ:

- text-only generation;
- image-conditioned text generation;
- synchronous endpoints cho immediate requests;
- asynchronous jobs cho queued work;
- client Python và Node.js dùng cùng API contract.

### 4. Hành vi service theo hướng production

Repository không chỉ có code load model. Nó còn bao gồm:

- API-key authentication;
- request IDs;
- protected worker restart bằng restart secret;
- queue limits và TTL behavior;
- health/readiness handling;
- sync/async generation routes;
- operational helpers cho Kaggle sessions;
- behavior-focused unit tests và contract tests.

### 5. Stable hot-request execution path

Dedicated compatible-request qualification ghi nhận:

```text
Warm request             38.651967787 s
Identical hot request     5.767615575 s
Speedup                   6.701550629×
Model load count          1
Same model object         True
Same output               True
Hot compile attempts      0
Hot compile events        0
Hot effective compile     0
OOM                       0
```

Điểm quan trọng không chỉ là hot request nhanh hơn. Cùng một logical model vẫn được giữ trong memory và compatible repeated request đã tái sử dụng stable decode execution path mà không quan sát thấy compile event mới.

## Runtime đã qualification

```text
Model source       keras/gemma4/Keras/gemma4_instruct_31b/2
Backend            JAX
Keras              3.15.1
KerasHub           0.32.0
JAX                0.10.2
JAXLIB             0.10.2
libtpu             0.0.49
Accelerator        Kaggle TPU v5e-8
TPU devices        8
Mesh               [1, 8]
Axes               [batch, model]
Dtype              bfloat16
```

## Public Kaggle production showcase

Canonical executed notebook:

**https://www.kaggle.com/code/dangkhoa2016/gemma-4-31b-kaggle-tpu-v5e-8-production-showcase**

Public notebook **Version 1** giữ lại một executed TPU v5e-8 showcase với:

- **6 text requests**;
- **5 vision requests**;
- **11 completed requests**;
- **24/24 executed code cells**;
- **23 output-bearing code cells**;
- **0 cell errors**;
- `SHOWCASE=PASS`;
- `COMPLETED_REQUESTS=11`.

Vision workload kết hợp các deterministic visual checks với một photorealistic synthetic portrait để kiểm tra cả structured visual reasoning lẫn natural-scene/attribute grounding.

Run-specific showcase timings đã ghi nhận:

```text
Server startup                 1920.829 s
Mean text inference              16.244 s
Mean vision inference            15.175 s
Identical-text hot speedup        3.728×
```

Các timing này được giữ như evidence của chính run đó, không được trình bày như một performance guarantee.

## Reproducibility: source template và executed evidence

Dự án cố ý duy trì hai notebook surface bổ sung cho nhau.

### GitHub source notebook

[`notebooks/kaggle-tpu-v5e8-text-vision-production-showcase.ipynb`](notebooks/kaggle-tpu-v5e8-text-vision-production-showcase.ipynb)

Vai trò:

- reproduction template sạch và dễ inspect;
- phù hợp để import vào Kaggle;
- mô tả exact setup và expected PASS markers;
- pin `RUNTIME_REF = 'v1.0.0'` để những lần chạy sau không vô tình chạy theo một `main` branch đã thay đổi trong tương lai.

### Kaggle executed notebook

Vai trò:

- public execution evidence;
- giữ real text và vision outputs;
- giữ completed-run timing/runtime metadata;
- cho phép reviewer so sánh run của họ với một executed reference.

Hai notebook này không cần byte-identical. Kaggle reserialize notebook JSON, còn independent reruns có thể khác về wall-clock timing và wording ở generated output. Reviewer nên so sánh model variation, runtime tag, workflow, code logic, PASS markers, request classes và semantic behavior.

## Tái lập trên Kaggle

Reviewer có thể làm theo quy trình sau:

1. Mở GitHub source notebook.
2. Import vào Kaggle bằng **Copy & Edit**.
3. Bật **TPU v5e-8**.
4. Bật Internet.
5. Attach chính xác:
   `keras/gemma4/Keras/gemma4_instruct_31b/2`
6. Kiểm tra model input mount.
7. Bắt đầu từ clean session nếu accelerator/model input vừa thay đổi.
8. Run All.
9. Xác nhận expected markers và real outputs.
10. So sánh behavior với public executed Version 1 notebook và release artifacts.

Expected model mount:

```text
/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2
```

Expected markers:

```text
TPU_CONFIG_IMPORTED=PASS
TPU_PREFLIGHT=PASS
SERVER_READY=PASS
SHOWCASE=PASS
COMPLETED_REQUESTS=11
```

## Khởi chạy nhanh

```bash
git clone https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision.git
cd KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision

cp .env.example .env
# Điền API_KEY và RESTART_SECRET vào .env.

bash scripts/start.sh
python scripts/wait_ready.py
```

Để chạy trên Kaggle TPU, hãy làm theo [docs/kaggle-notebook.vi.md](docs/kaggle-notebook.vi.md) thay vì chỉ dựa vào shell quick-start ở trên.

## Ví dụ API

### Text generation

```bash
curl -sS http://127.0.0.1:7860/generate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Giải thích TPU model parallelism trong ba câu.",
    "max_new_tokens": 96
  }'
```

### Vision-conditioned generation

Image-conditioned generation được expose qua:

```text
/generate/image
/generate/image/async
```

Async API phù hợp với các workflow không muốn giữ một HTTP request mở trong suốt thời gian request đang queue hoặc generate.

## Release evidence

GitHub Release `v1.0.0` chứa các reviewer-facing artifacts của executed public showcase:

- `gemma4-v1.0.0-public-showcase-evidence.zip`
- `kaggle-tpu-v5e8-production-executed-public.html`
- `kaggle-tpu-v5e8-production-executed-public.ipynb`
- `showcase-results-public.json`

Canonical release provenance được neo bằng annotated tag `v1.0.0` và GitHub Release. Exact release commit, source tree, artifact checksums và reviewer-facing evidence được công bố tại đó để có thể đối chứng chính xác mà không tạo self-referential commit hash ngay bên trong README này.

Release evidence được thiết kế để project có thể được review mà không yêu cầu người đọc phải tin vào screenshot hoặc undocumented local run.

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
.github/      GitHub workflows, templates và repository policy
clients/      client Python và Node.js
docs/         tài liệu public song ngữ
evidence/     qualification summary public
notebooks/    Kaggle production notebook
scripts/      operational và verification helpers
src/          production server và TPU runtime
tests/        behavior-focused unit/contract tests
```

## Phạm vi và giới hạn

Dự án chứng minh một engineering path đã được validation cho runtime và workload đã ghi nhận. Dự án không tuyên bố:

- universal latency cho mọi prompt hoặc mọi Kaggle session trong tương lai;
- SLA;
- token-for-token output deterministic cho mọi rerun;
- arbitrary Gemma variants sẽ chạy mà không cần requalification;
- endorsement từ Google, Kaggle, Keras hay OpenAI.

Xem thêm [docs/limitations.vi.md](docs/limitations.vi.md) để biết các operational constraints chi tiết.

## Release

Production release hiện tại: **[v1.0.0](https://github.com/dangkhoa2016/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision/releases/tag/v1.0.0)**.

Release này là stable baseline mà reproduction notebook và public evidence workflow sử dụng.

## License

MIT. Xem [LICENSE](LICENSE) và [NOTICE](NOTICE).
