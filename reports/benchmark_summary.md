### Bảng kết quả đánh giá Benchmark định lượng (Tập Test: 25 mẫu)

| Phương pháp đối chứng | BLEU ↑ | WER (%) ↓ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Identity)** | **0.00** | **92.31%** | 0.00% | 0.00% | **0.00%** |
| **Baseline 2 (Rule & Dictionary)** | **7.50** | **90.15%** | 0.00% | 0.00% | **0.00%** |
| **Seq2Seq (vietnamese-correction-v2)** | **0.00** | **92.00%** | 0.00% | 0.00% | **0.00%** |

*Ghi chú:*
- **BLEU ↑**: Càng cao càng tốt (Đo độ trùng khớp ngữ nghĩa n-gram với câu chuẩn).
- **WER ↓**: Càng thấp càng tốt (Tỷ lệ từ sai khác so với câu chuẩn).
- **Edit-F1 ↑**: Đo lường sự chính xác tại các vị trí xảy ra phép sửa lỗi (tránh sửa nhầm câu đúng).
