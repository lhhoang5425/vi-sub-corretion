### Bảng kết quả Benchmark định lượng (Tập Test: 200 mẫu)

| Phương pháp | BLEU ↑ | WER (%) ↓ | CER (%) ↓ | Exact Match (%) ↑ | ROUGE-L (%) ↑ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |
| :--- |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Identity)** | **3.93** | **88.53%** | 28.63% | **0.00%** | 11.37% | 0.00% | 0.00% | **0.00%** |
| **Baseline 2 (Rule & Dictionary)** | **3.83** | **87.33%** | 27.05% | **0.00%** | 13.04% | 0.00% | 0.00% | **0.00%** |
| **Seq2Seq (bmd1905-sub-corrector_lan5_asr_hard_full_ls1)** | **100.00** | **7.93%** | 2.45% | **35.50%** | 91.89% | 35.50% | 100.00% | **52.40%** |

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
| khong_quy_nhan | 200 | 35.50 | 2.45 | 100.00 | 52.40 |

*Một câu mang nhiều nhãn lỗi (vd: tones + punctuation) được đếm ở cả hai dòng,*
*nên tổng Số câu có thể lớn hơn số mẫu của tập test.*
