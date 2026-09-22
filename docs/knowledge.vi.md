# Kiến thức kỹ thuật

## Vì sao một model trên 8 TPU devices?

Model 31B quá lớn để xem mỗi TPU device như một replica độc lập trong memory
envelope mục tiêu. Keras ModelParallel shard các parameter lớn theo model axis
nhưng vẫn giữ một logical model.

## Vì sao worker phải sống lâu?

Load model và JAX compile đắt. Worker sống lâu giữ model state và cho phép hot
request có shape tương thích tái sử dụng executable đã compile.

## Vì sao callable identity quan trọng?

JAX compilation cache nhạy với Python callable identity. Nếu tạo lại nested
`lax.while_loop` callable cho từng request, JAX có thể retrace/recompile. Sampler
production giữ callable ổn định và đưa dữ liệu riêng của request vào loop state.

## Vì sao không dùng outer `run_eagerly=False`?

Cách này đã được thử trên runtime 31B nhưng tạo host-memory pressure rất lớn khi
PJRT materialize sharding/layout metadata. Stable inner loop đạt reuse mà không
gặp behavior đó.
