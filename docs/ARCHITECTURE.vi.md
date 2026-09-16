# Kiến trúc

```text
HTTP
 |
 v
Flask + Waitress CPU coordinator
 |
bounded multiprocessing queue + JobStore
 |
ONE spawned TPU worker
 |
Gemma4TPUEngine
 |
Keras ModelParallel mesh [1,8]
 |
TPU0 ... TPU7
```

Đây là một logical model shard qua 8 TPU, không phải 8 replicas.

JAX/Keras/KerasHub chỉ import trong spawned worker sau khi TPU environment đã
được cấu hình. Candidate A shard dominant dense text weights và để vision
encoder replicated cho tới khi strict-load feasibility được proven.

G3/G5 ban đầu được đánh dấu `keras_hub_native_unvalidated`; G4 có thể thay bằng
Gemma4-native split generation sau real TPU characterization.
