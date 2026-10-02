# Kaggle Production Showcase

Notebook public chính thức: [https://www.kaggle.com/code/dangkhoa2016/gemma-4-31b-kaggle-tpu-v5e-8-production-showcase](https://www.kaggle.com/code/dangkhoa2016/gemma-4-31b-kaggle-tpu-v5e-8-production-showcase)

## Môi trường khuyến nghị

1. Tạo Kaggle notebook.
2. Chọn TPU v5e-8 / v5litepod-8.
3. Bật Internet khi cần.
4. Attach Gemma 4 31B Keras model preset.
5. Mở `notebooks/kaggle-tpu-v5e8-text-vision-production-showcase.ipynb` và chọn **Run All**.

## Notebook trình diễn những gì

Notebook được thiết kế rộng hơn một smoke test. Một lần showcase đầy đủ gồm:

- TPU preflight và metadata runtime đang chạy;
- sáu request dạng text/chat, gồm tiếng Anh, tiếng Việt, sinh code, tuân thủ
  cấu trúc và một identical hot repeat;
- một visual diagnostic card phong phú hơn được tự tạo bằng Pillow;
- ba image-conditioned request trên diagnostic card để kiểm tra hiểu cảnh,
  định vị/đếm và đọc chữ/bảng;
- hai image-conditioned request trên chân dung photorealistic synthetic do
  Mage-Flow-Turbo sinh để kiểm tra hiểu cảnh tự nhiên và attribute grounding;
- ảnh riêng tùy chọn qua biến `SHOWCASE_IMAGE`;
- wall time, inference time, generation bucket và compile-cache evidence cho
  từng request;
- bản tổng hợp machine-readable tại
  `/kaggle/working/gemma4-showcase-results.json`.

Gemma 4 trong dự án này nhận ảnh và sinh văn bản. Đây **không phải** model
text-to-image.

## Cách hiểu hiệu năng

Load model và compile lần đầu rất tốn thời gian. Cần giữ worker sống để các
request tương thích có thể tái sử dụng compiled executable. Qualification trước
đó ghi nhận khoảng 541,4 giây cho request tương thích đầu tiên và khoảng 5,24
giây cho identical hot request. Showcase sẽ đo lại trên chính các prompt của nó;
không xem các số qualification này là SLA.

## Persistence

Có thể dùng `/kaggle/working` cho project state, visual test card được tạo, JSON
kết quả showcase và JAX compilation cache theo cơ chế persistence của Kaggle.
Accelerator và kernel đang chạy vẫn nên được xem là ephemeral.

## Ảnh riêng tùy chọn

Sau phần vision demo xác định, đặt `SHOWCASE_IMAGE` tới file `.jpg`, `.jpeg`,
`.png` hoặc `.webp` dưới `/kaggle/input` để chạy thêm ảnh thật của bạn.

## Public endpoint

Quick Tunnel vẫn là tùy chọn. `scripts/run_tunnel.sh` forward local server mà
không cần Named Tunnel hay custom domain.