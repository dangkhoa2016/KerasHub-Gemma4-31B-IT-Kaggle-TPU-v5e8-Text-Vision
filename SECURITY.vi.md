# Chính sách bảo mật

## Báo cáo

Không đăng bí mật, thông tin đăng nhập, API key, restart secret, token mô hình
hay URL notebook riêng tư trong issue công khai.

Nếu bạn phát hiện một lỗ hổng, hãy mở một issue kèm mô tả tối thiểu, không bao
gồm bí mật, và cho biết cách tái hiện.

## Cấu hình an toàn

- Giữ `.env` ngoài kiểm soát phiên bản; `.env.example` chỉ chứa giá trị mẫu.
- Bắt buộc xác thực API key trên mọi endpoint được mở ra ngoài.
- Dùng restart secret riêng, không dùng lại API key.
- Xem Quick Tunnel là cổng vào minh họa, không phải biên giới an toàn lâu dài.
- Xoay vòng mọi thông tin xác thực có thể đã lộ trong log hoặc ảnh chụp màn
  hình.
- Không commit tệp `.env`, cache biên dịch JAX, PID hay log server.

## Phạm vi

Phát hiện trong `src/gemma4_server/` nằm trong phạm vi của chính sách này. Vấn
đề tài liệu hoặc mẫu không nhạy cảm nên dùng quy trình issue thông thường.
