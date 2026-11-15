> ⚠️ **CẢNH BÁO NGUỒN GỐC — ĐỌC TRƯỚC KHI TRÍCH DẪN.**
> Bảng dưới đây **KHÔNG** phải kết quả của mô hình Lần 4. Hàng `Seq2Seq` ở đây (BLEU 0.00 / WER 92.00%)
> là kết quả đo **mô hình gốc chưa fine-tune** trong một lần chạy local, gần trùng Baseline 1 Identity
> (92.31%) — nghĩa là mô hình chỉ thêm viết hoa và dấu câu, giữ nguyên toàn bộ phần chữ.
> Nguyên nhân: biến môi trường `MODEL_NAME_OR_PATH` đặt ở cấp hệ thống đè lên tệp `.env`.
>
> **Số liệu đúng của mô hình Lần 4 (BLEU 27.09 / WER 14.77%):** xem [`benchmark_lan4.md`](./benchmark_lan4.md).
>
> Tệp này do `run_benchmark.py` tự động ghi đè mỗi lần chạy. Bản đóng băng theo từng lần train nằm ở
> `reports/benchmark_lan<N>.md`.

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
