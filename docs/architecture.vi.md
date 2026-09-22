# Kiến trúc

```text
HTTP
 |
 v
Flask + Waitress coordinator
 |
bounded queue + JobStore
 |
một TPU worker sống lâu
 |
Gemma4TPUEngine
 |
Keras ModelParallel mesh [1,8]
 |
8 TPU devices
```

Hệ thống chạy một logical Gemma 4 31B model được shard trên 8 TPU devices,
không phải 8 replica độc lập.

## Load model

JAX, Keras và KerasHub được cấu hình bên trong spawned worker. Model được load
strict qua KerasHub preset loader trong ModelParallel scope. Temporary host
allocation được GC và trim sau load.

## Generation

KerasHub giữ native prefill/cache semantics. Greedy sampler production giữ
identity của JAX `lax.while_loop` ổn định giữa các request, còn prompt, cache,
padding mask, stop IDs và variable values vẫn là dynamic state. Nhờ vậy hot
request có shape tương thích có thể tái sử dụng compiled executable.

## Vision

Image conditioning được tạo ở bước preprocess/prefill của KerasHub. Decode loop
dùng cache đã được condition và không thay vision encoder gốc.
