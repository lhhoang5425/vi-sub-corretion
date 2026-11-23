### Bảng kết quả Benchmark định lượng (Tập Test: 132 mẫu)

| Phương pháp | BLEU ↑ | WER (%) ↓ | CER (%) ↓ | Exact Match (%) ↑ | ROUGE-L (%) ↑ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |
| :--- |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Identity)** | **79.66** | **6.36%** | 3.91% | **54.55%** | 93.95% | 0.00% | 0.00% | **0.00%** |
| **Baseline 2 (Rule & Dictionary)** | **79.66** | **8.59%** | 4.43% | **35.61%** | 91.48% | 0.00% | 0.00% | **0.00%** |
| **Seq2Seq (bmd1905-sub-corrector_lan5_asr_hard_full_ls1)** | **79.66** | **9.61%** | 5.57% | **34.85%** | 90.69% | 3.45% | 6.25% | **4.44%** |

*Ghi chú chỉ số:*
- **BLEU / ROUGE-L ↑**: độ tương đồng với câu chuẩn (BLEU thiên về n-gram khớp hẳn, ROUGE-L thiên về bao phủ).
- **WER ↓ / CER ↓**: tỷ lệ lệch ở mức từ và mức KÝ TỰ. CER mới thấy được lỗi thiếu dấu thanh,
  vì WER vẫn đếm một từ sai dấu là đúng 1 từ sai dù chỉ lệch một ký tự.
- **Exact Match ↑**: tỷ lệ câu trùng 100% đáp án chuẩn. Con số dễ đọc nhất cho hội đồng.
- **Edit-P ↑**: trong các câu mô hình đụng vào sửa, bao nhiêu % sửa đúng đích (càng thấp càng over-correct).
- **Edit-R ↑**: bao nhiêu % câu cần sửa thì được can thiệp (100% = không bỏ sót câu nào).
- **Edit-F1 ↑**: trung bình điều hòa của Edit-P và Edit-R.


### Bảng hiệu năng theo loại lỗi (Seq2Seq (bmd1905-sub-corrector_lan5_asr_hard_full_ls1))

| Loại lỗi | Số câu | Exact Match (%) ↑ | CER (%) ↓ | BLEU ↑ | Edit-F1 (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| asr_real | 132 | 34.85 | 5.57 | 79.66 | 4.44 |
| low_confidence | 19 | 0.00 | 14.38 | 67.61 | 0.00 |

*Một câu mang nhiều nhãn lỗi (vd: tones + punctuation) được đếm ở cả hai dòng,*
*nên tổng Số câu có thể lớn hơn số mẫu của tập test.*
