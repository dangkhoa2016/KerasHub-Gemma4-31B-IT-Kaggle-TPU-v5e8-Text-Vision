# Nhật ký thay đổi

Mọi thay đổi có ảnh hưởng tới người dùng công khai đều được ghi lại tại đây.

## Chưa phát hành

Chưa có thay đổi công khai nào được ghi nhận sau v1.0.0.

## v1.0.0 — 2026-09-26

### Tài liệu và hình thức kho mã

- Bổ sung trang chủ và tài liệu song ngữ tiếng Anh / tiếng Việt.
- Bổ sung mẫu issue, hướng dẫn pull request, cấu hình Dependabot và bản tóm
  tắt chứng nhận công khai.
- Đơn giản hóa cây mã nguồn công khai bằng cách gỡ các artifact gate và
  runbook phát triển nội bộ.
- Đổi tên notebook, manifest phụ thuộc và lớp kiểm thử theo hành vi mà người
  dùng quan sát được thay vì nhãn chứng nhận nội bộ.
- Bổ sung bản tiếng Việt cho README của client, changelog, hướng dẫn đóng góp
  và chính sách bảo mật.
- Viết lại production notebook theo workflow công khai của repository và bổ
  sung static validation cho tham chiếu notebook, local link, cặp tài liệu
  song ngữ và thuật ngữ phát triển nội bộ.

### Runtime

- Tái sử dụng JAX decode-loop callable ổn định cho các hot request tương thích,
  đồng thời giữ nguyên ngữ nghĩa prefill/cache gốc của KerasHub.
- Giữ nguyên strict checkpoint loading, ModelParallel sharding, text
  generation, image-conditioned generation, REST API có xác thực và async job.
- Công bố metadata runtime đã được qualification trong API và tài liệu.
- Đồng bộ TPU dependency manifest với runtime đã được qualification.
- Qualification KerasHub 0.32.0 và libtpu 0.0.49, đồng thời tăng cold model-load timeout lên 3600 giây để checkpoint 60 GB có đủ thời gian load trên Kaggle mount còn lạnh.

### Qualification

- Qualification production source trên Kaggle TPU v5e-8 với một lần load model.
- Ghi nhận request tương thích đầu tiên 38.652 giây và hot request giống
  hệt 5.768 giây (warm-to-hot speedup 6.70x) mà không phát sinh compile event mới.
- Công bố tài liệu qualification song ngữ gọn và evidence machine-readable.
