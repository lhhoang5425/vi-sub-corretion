# `data/gold_test/` — Các tập test (đề thi + đáp án)

> ⚠️ **Đừng lẫn với `tests/` ở gốc dự án.** Hai thứ khác hẳn nhau:
> - **`tests/`** = unit test của **mã nguồn** (chạy `pytest` sau khi sửa code)
> - **`data/gold_test/`** (thư mục này) = tập dữ liệu để **đo mô hình**
>
> Trùng chữ "test" nhưng một cái kiểm *code*, một cái kiểm *model*.

---

## Cách dùng trong một câu

```bash
python -m src.evaluation.run_benchmark \
    --test_file data/gold_test/<thư-mục>/<tập>.jsonl \
    --model_name_or_path <đường-dẫn-checkpoint> \
    --output_json <nơi-ghi-kết-quả>.json
```

Tập test **không** dùng để train. Train dùng `data/synthetic/train_*.jsonl` (sinh bởi `corpus_builder`).

---

## Cấu trúc

```
data/gold_test/
├── gd1_basic/        GIAI ĐOẠN 1 — bài toán gốc mà đề cương đã tuyên bố
├── gd2_real_asr/     GIAI ĐOẠN 2 — ASR thật (Whisper), mức nâng cao
└── control/          ĐỐI CHỨNG — đo over-correction & ngoài miền huấn luyện
```

---

## `gd1_basic/` — Giai đoạn 1 (báo cáo giảng viên)

Đầu vào là văn bản **mất dấu thanh hoàn toàn**. Đây là bài toán Mục 1–2 đề cương mô tả.

| Tập | Câu | Đặc điểm | Dùng khi nào |
| :--- | ---: | :--- | :--- |
| `sample_test.jsonl` | 25 | Bản gốc. 100% mất dấu, mất dấu câu, viết thường | **Chỉ để đối chiếu** con số đã báo (WER 14,77%) |
| **`basic_test_unaccent_all.jsonl`** | **200** | Cùng dạng 100% mất dấu, nhưng gấp 8 lần mẫu | **Tập chính của Giai đoạn 1** |
| `basic_test_asr_hard.jsonl` | 200 | Nhiễu hỗn hợp (76% mất trắng + 10% câu sạch) | Kiểm khả năng chịu nhiều mức lỗi |

**Vì sao có tập cơ bản:** 25 câu quá nhỏ — mỗi câu chiếm 4% chỉ số, nên không phân biệt
được cải thiện thật với nhiễu đo. Tập 200 câu giảm mỗi câu còn 0,5%.

Hai tập `basic_test_*` được sinh bởi `python -m src.data_processing.build_basic_test`,
câu lấy từ vùng **chưa từng dùng để train** (anti-join theo nội dung).

---

## `gd2_real_asr/` — Giai đoạn 2 (nâng cao)

Đầu vào là **phụ đề Whisper thật**, chạy trên video thật. KHÁC hẳn Giai đoạn 1: Whisper
xuất ra văn bản **có** dấu thanh (99,3%), **có** dấu câu (81,1%), **có** viết hoa (82,4%).
Lỗi thật của nó là **sai dấu** và **nhầm phụ âm đầu** — không phải mất dấu.

| Tập | Câu | Dùng khi nào |
| :--- | ---: | :--- |
| `kichban_test.jsonl` | 132 | Chạy benchmark mô hình train bằng hồ sơ `whisper_like` |
| `kichban_review.md` | — | Bảng để **người duyệt** bản gán nhãn (60 câu có sửa, 19 câu cần nghe audio) |

Nguồn: `data/raw/kichban_2026-10-25.srt`, sinh bởi `build_kichban_test.py`.
16 block cuối (quảng cáo sản phẩm người lớn) đã bị loại khỏi tập test.

---

## `control/` — Đối chứng

Dùng để đo **over-correction** (câu vốn đã đúng mà mô hình sửa nhầm) và **ngoài miền
huấn luyện** (văn học dịch, câu dài ~35 từ, khác hẳn câu hội thoại của tập train).

| Tập | Câu | Đặc điểm | Đo cái gì |
| :--- | ---: | :--- | :--- |
| **`corpus_identity.jsonl`** | **962** | Đầu vào = đầu ra = câu đã đúng | **Mọi thay đổi đều là sửa nhầm** |
| `corpus_noisy.jsonl` | 962 | Sinh nhiễu từ cùng câu | Khôi phục trên nội dung lạ |
| `corpus_test_manifest.json` | — | Hash + thống kê | Chứng minh tập test tái lập được |

**Vì sao cần:** tập test cũ **không có câu sạch nào**, nên nhóm "câu vốn đúng bị sửa nhầm"
luôn rỗng — hiện tượng `"tổ buôn 247" → "tổ chức 247"` ở Mục 7 đề cương **chưa từng đo được**.

Nguồn: `corpus/*.txt` (văn học dịch). **Tệp văn bản và `corpus_*.jsonl` KHÔNG được commit
lên git** vì lý do bản quyền — xem `.gitignore`. Chỉ script sinh và manifest (hash) được commit.

---

## Bảng tra nhanh

| Muốn… | Dùng tập |
| :--- | :--- |
| Báo cáo giảng viên (bài toán gốc) | `gd1_basic/basic_test_unaccent_all.jsonl` |
| Đối chiếu với con số 14,77% đã công bố | `gd1_basic/sample_test.jsonl` |
| Kiểm over-correction (mô hình có phá câu đúng không) | `control/corpus_identity.jsonl` |
| Kết quả nâng cao (ASR thật) | `gd2_real_asr/kichban_test.jsonl` |

## Quy tắc khi tạo tập test mới

1. Câu phải **chưa từng** dùng để train — anti-join theo **nội dung**, không theo index
   (dataset ShynBui trùng lặp nội bộ ~21%).
2. Nhiễu phải sinh bằng **hồ sơ khác** cấu hình đã dùng để train, nếu không sẽ đo
   "mô hình thuộc lòng hàm nhiễu nhà mình".
3. Chốt một lần, commit, **đóng băng**. Không sinh lại sau khi đã thấy kết quả.
