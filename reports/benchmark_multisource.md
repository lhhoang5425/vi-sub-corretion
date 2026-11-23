### Bảng kết quả Benchmark định lượng (Tập Test: 300 mẫu)

| Phương pháp | BLEU ↑ | WER (%) ↓ | CER (%) ↓ | Exact Match (%) ↑ | ROUGE-L (%) ↑ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |
| :--- |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Identity)** | **4.11** | **89.20%** | 28.17% | **0.00%** | 11.14% | 0.00% | 0.00% | **0.00%** |
| **Baseline 2 (Rule & Dictionary)** | **3.98** | **87.99%** | 26.59% | **0.00%** | 12.72% | 0.00% | 0.00% | **0.00%** |
| **Seq2Seq (bmd1905-sub-corrector_lan5_asr_hard_full_ls1)** | **100.00** | **16.20%** | 6.28% | **20.67%** | 83.74% | 20.67% | 100.00% | **34.25%** |

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
| khong_quy_nhan | 300 | 20.67 | 6.28 | 100.00 | 34.25 |

*Một câu mang nhiều nhãn lỗi (vd: tones + punctuation) được đếm ở cả hai dòng,*
*nên tổng Số câu có thể lớn hơn số mẫu của tập test.*


### Bảng hiệu năng theo nguồn dữ liệu (Seq2Seq (bmd1905-sub-corrector_lan5_asr_hard_full_ls1))

| Nguồn | Số câu | Exact Match (%) ↑ | CER (%) ↓ | BLEU ↑ | Edit-F1 (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| shynbui | 150 | 36.67 | 2.59 | 100.00 | 53.66 |
| wikipedia | 150 | 4.67 | 9.89 | 83.77 | 8.92 |

*Chênh lệch lớn giữa các nguồn nghĩa là mô hình chưa tổng quát hoá,*
*mà chỉ khớp văn phong của nguồn chiếm đa số trong tập train.*
