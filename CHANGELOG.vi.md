# Nhật ký thay đổi

Mọi thay đổi có ảnh hưởng tới người dùng công khai đều được ghi lại tại đây.

## Chưa phát hành

### Tài liệu và hình thức kho mã

- Bổ sung trang chủ và tài liệu song ngữ tiếng Anh / tiếng Việt.
- Bổ sung mẫu issue, hướng dẫn pull request, cấu hình Dependabot và bản tóm
  tắt chứng nhận công khai.
- Đơn giản hóa cây mã nguồn công khai bằng cách gỡ các tệp quy trình phát
  triển nội bộ.
- Đổi tên notebook, manifest phụ thuộc và lớp kiểm thử theo hành vi mà người
  dùng quan sát được, thay vì nhãn chứng nhận nội bộ.
- Bổ sung bản tiếng Việt cho `clients/README.md`, `CHANGELOG.md`,
  `CONTRIBUTING.md` và `SECURITY.md`.
- Viết lại notebook production theo quy trình công khai chỉ dùng các tệp có
  thật trong kho mã.
- Bổ sung trình kiểm tra cây công khai vào CI cho tham chiếu tệp, thuật ngữ
  nội bộ và cặp tài liệu song ngữ.

### Runtime

- Tái sử dụng vòng lặp giải mã JAX ổn định cho các yêu cầu nóng tương thích,
  giữ nguyên ngữ nghĩa prefill/cache gốc của KerasHub.
- Giữ nguyên nạp checkpoint nghiêm ngặt, sharding ModelParallel, sinh văn bản,
  sinh văn bản có điều kiện hình ảnh, REST API có xác thực và job bất đồng bộ.
- Công bố metadata runtime đã chứng nhận trong API và tài liệu.
- Căn `requirements-tpu.txt` về đúng phiên bản runtime đã chứng nhận.

## v1.0.0 — dự kiến

Bản phát hành công khai đầu tiên sẽ đóng gói runtime Gemma 4 31B trên Kaggle
TPU đã chứng nhận, sau khi hoàn tất rà soát phát hành cuối cùng. Chính mục
nhật ký này tự thân không tạo ra thẻ `v1.0.0` công khai.
