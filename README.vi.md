# KerasHub Gemma 4 31B Instruct trên Kaggle TPU v5e-8 — Text + Vision

Đây là **source tree độc lập mới**, kế thừa các bài học kiến trúc từ project
TranslateGemma 27B đã hoàn tất.

> **G8 closeout cuối:** qualification CPU async/lifecycle và live generation authority cuối đã đóng. Evidence bất biến nằm trong `artifacts/g8/`, gồm post-fix và formal closeout packages.

## Trạng thái

```text
G0 standalone source tree                    CLOSED
G1 TPU/model preflight                       CLOSED
R3 sharded checkpoint assignment              CLOSED/PASS
G2 strict checkpoint load                    CLOSED/PASS
G3 text generation                           CLOSED/PASS
G4 generation architecture characterization CLOSED/PASS (SPLIT)
G5 image generation                          CLOSED/PASS (NATIVE_VISION)
G6 final sharding + memory evidence          CLOSED/PASS
G7 REST server                               CLOSED/PASS (CPU REST contract)
G8 async/cold compile/lifecycle              CLOSED/PASS
G8 live generation                           PASS
G9 PRIME/HOT                                 ĐỦ ĐIỀU KIỆN / CHƯA BẮT ĐẦU
G10 Restart Session mới -> Run All           CHƯA BẮT ĐẦU
G11 history/source hardening                 ĐANG CHUẨN BỊ CPU
G12 public v1.0.0                            CHƯA RELEASE
```

Phần vận hành còn lại được nén thành hai phase:

```text
STEP 1  FINAL CPU PREP       chuẩn bị source/history/release
STEP 2  FINAL TPU ONE-SHOT   kiểm chứng G9 PRIME/HOT và G10 session mới
```

G8 cuối ghi nhận `G9_ENTRY_ELIGIBLE=true` và `G9_STARTED=false`. PRIME/HOT,
fresh-session validation, tag và release vẫn chưa được thực hiện.

Gemma3/TranslateGemma split engine cũ không được copy mù quáng. G3/G5 ban đầu
chỉ dùng native Gemma 4 generation để characterization. Nếu real TPU evidence
cho thấy compile graph hoặc host memory không an toàn, G4 sẽ chuyển sang split
engine Gemma4-native.

Chạy đầu tiên:

```bash
cp .env.example .env
bash scripts/run_g0_g2.sh
```

Chỉ sau khi `artifacts/g0-g2/strict-load.json` PASS mới mở G3/G5 và REST.
