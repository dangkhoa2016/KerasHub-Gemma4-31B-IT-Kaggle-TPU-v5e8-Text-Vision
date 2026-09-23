# Kaggle Notebook

## Môi trường khuyến nghị

1. Tạo Kaggle notebook.
2. Chọn TPU v5e-8 / v5litepod-8.
3. Bật Internet khi cần.
4. Attach Gemma 4 31B Keras model preset.
5. Clone hoặc upload repo vào `/kaggle/working`.
6. Cài requirements, cấu hình secret và chạy server.

Production notebook nằm trong `notebooks/`.

## Persistence

Có thể dùng `/kaggle/working` cho project state và JAX compilation cache theo
cơ chế persistence của Kaggle. Accelerator allocation và kernel đang chạy vẫn
nên được xem là ephemeral.

## Compile đầu tiên

Load Gemma 4 31B và compile JAX đầu tiên rất tốn thời gian. Hãy giữ worker sống
để request hot có shape tương thích tái sử dụng executable.
