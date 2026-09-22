# Release Notes v1.0.0

Bản public đầu tiên tập trung vào stack inference Gemma 4 31B Instruct có thể
tái lập trên Kaggle TPU v5e-8.

## Bao gồm

- strict preset loading qua KerasHub
- Keras ModelParallel sharding trên 8 TPU devices
- BF16 runtime
- text generation và image-conditioned text generation
- stable JAX decode-loop reuse cho hot request tương thích
- REST API sync + async
- API key, request ID và worker restart có bảo vệ
- client Python và Node.js
- script vận hành Kaggle và production notebook
- tài liệu song ngữ Anh / Việt
- qualification evidence gọn

## Qualification

Production-source qualification dùng một model load. Request tương thích đầu
tiên chịu compile cost; request hot giống hệt tái sử dụng executable và không
phát sinh compile event mới. Xem [qualification.vi.md](qualification.vi.md).

## Phạm vi

Release này nhắm tới Kaggle TPU v5e-8 / v5litepod-8. Accelerator khác và
dependency version tương lai không nằm trong cùng qualification.
