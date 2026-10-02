# Đóng góp

Chào mừng mọi đóng góp giữ cho dự án tập trung vào suy luận Gemma 4 có thể
tái lập được và vận hành rõ ràng.

## Trước khi mở pull request

1. Mở issue trước với mọi thay đổi hành vi lớn.
2. Giữ mỗi commit tập trung và mô tả rõ ý định mà người dùng quan sát được.
3. Chạy `bash scripts/test_unit.sh`.
4. Cập nhật đồng thời tài liệu tiếng Anh và tiếng Việt khi hành vi công khai
   thay đổi.
5. Không bao giờ commit API key, token notebook, thông tin đăng nhập mô hình
   hay URL riêng tư.
6. Chạy `python3 scripts/validate_public_tree.py` để kiểm tra cặp tài liệu
   song ngữ và các tham chiếu tệp trong notebook.

## Phạm vi thay đổi

- Runtime đã được chứng nhận phải giữ nguyên. Không sửa `src/gemma4_server/`
  nếu không có lỗi runtime thật kèm bằng chứng tái lập được.
- Manifest phụ thuộc chỉ được căn theo runtime đã chứng nhận, không nâng cấp
  phiên bản mới chỉ vì Dependabot đề xuất.
- Không thêm thuật ngữ nội bộ của quá trình phát triển vào tài liệu công khai.

## Hiệu năng và khả năng tương thích trên TPU

Mọi tuyên bố về hiệu năng hoặc khả năng tương thích trên TPU phải kèm chi tiết
runtime tái lập được và bằng chứng đo được. Xem
[docs/qualification.md](docs/qualification.md).

## Ngôn ngữ

Nội dung tài liệu công khai phải có cả bản tiếng Anh và tiếng Việt trong cùng
một commit. Chạy `python3 scripts/validate_public_tree.py` để kiểm tra các cặp
tài liệu song ngữ hiện có trước khi mở pull request.