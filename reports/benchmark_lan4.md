# Benchmark Lần 4 — 100.000 câu

> **Bản ĐÓNG BĂNG.** Số liệu trích từ log Colab gốc `reports/benchmark_tay.txt`, không phải từ một lần chạy benchmark nào khác. Ngày train: 02/11/2026 · Thiết bị: Tesla T4 · fp16.

## 1. Cấu hình đợt train

| Thông số | Giá trị |
| :--- | :--- |
| Mô hình nền | `bmd1905/vietnamese-correction-v2` |
| Câu huấn luyện | 90.000 |
| Câu validation | 10.000 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Max length | 128 (pad **cứng** mọi câu) |
| Weight decay | 0.01 |
| Label smoothing | 0.0 (không dùng) |
| Thời gian train | 7.941 giây = **2 giờ 12 phút** |
| Tổng bước | 16.875 |
| `train_loss` | 0.03736 |
| `eval_loss` theo epoch | 0.02764 → 0.01632 → **0.01307** |
| Throughput | 34 câu/giây |

## 2. Kết quả đối đầu (tập test vàng — 25 mẫu)

| Phương pháp | BLEU ↑ | WER (%) ↓ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline 1 — Identity | 0.00 | 92.31 | 0.00 | 0.00 | 0.00 |
| Baseline 2 — Rule & Dictionary | 7.50 | 90.15 | 0.00 | 0.00 | 0.00 |
| **Seq2Seq (Lần 4)** | **27.09** | **14.77** | **28.00** | **100.00** | **43.75** |

**So với baseline:** WER giảm 92.31% → 14.77% (77,5 điểm phần trăm). BLEU 0.00 → 27.09.

## 3. Lưu ý khi đọc kết quả

1. **Tập test chỉ 25 mẫu.** Mỗi câu xấp xỉ 4% chỉ số, nên phương sai đo được lớn. Không kết luận được chênh lệch nhỏ hơn vài điểm.
2. **Một loại lỗi duy nhất:** cả 25 câu đều mất trắng dấu thanh. Nhãn theo `error_type`: `tones` 25/25, `punctuation` 25/25, `capitalization` 25/25, **`homophone` chỉ 1/25** — trong khi từ đồng âm là một trong bốn phạm vi cốt lõi ở Mục 1 đề cương.
3. **Edit-P 28,00%** nghĩa là 72% phép sửa không khớp đáp án. Đây **không** phải over-correction. Tập test không có câu sạch nào, nên phép đo over-correction luôn rỗng. Con số 28% đo tỷ lệ sửa trúng đích.
4. **`eval_loss` 0.01307 bị ảnh hưởng bởi rò rỉ** val↔train 20,8% do dataset nguồn trùng lặp nội bộ. Hướng giảm đơn điệu qua 3 epoch vẫn là bằng chứng chống overfitting, nhưng độ lớn tuyệt đối không đáng tin.
5. **Không đối chiếu với `reports/benchmark_summary.md`.** Tệp đó chứa số của một lần chạy khác — mô hình **gốc chưa fine-tune** (WER 92,00%), không phải mô hình Lần 4. Xem ghi chú ở đầu tệp đó.

## 4. Nguồn bằng chứng

| Tệp | Nội dung |
| :--- | :--- |
| `reports/benchmark_tay.txt` | Log Colab thô, chứa cả quá trình train và benchmark |
| `notebooks/seq2seq_train_4_100k.ipynb` | Output train đầy đủ, 2.459 dòng |
| Đoạn log đối chiếu | `{'eval_loss': '0.01307', 'epoch': '3'}` · `{'train_runtime': '7941', 'train_loss': '0.03736'}` · `-> BLEU: 27.09 \| WER: 14.77% \| Edit-F1: 43.75%` |

## 5. Trạng thái tái lập

⚠️ **Checkpoint Lần 4 chỉ còn trong `/content` — bộ nhớ tạm của Colab, nhiều khả năng đã bị thu hồi.** Trên Drive có `bmd1905_checkpoint.zip` (9,5 GB) nhưng nén kèm `checkpoint-*/` chứa optimizer state; cần giải nén và kiểm tra trước khi tin.

Cách khôi phục nhanh: dùng `Seq2SeqCorrector(model_name_or_path="bmd1905/vietnamese-correction-v2")` — nhưng đó là **mô hình gốc**, cho WER 92%, **không phải** mô hình đã đo ở bảng trên.
