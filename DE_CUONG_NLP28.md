# ĐỀ CƯƠNG ĐỀ TÀI BÀI TẬP NHÓM NLP

**Học phần:** Xử lý ngôn ngữ tự nhiên (NLP) — UNETI 2026  
**Giảng viên hướng dẫn:** Thầy Bùi Văn Tân  
**Mã đề tài:** `NLP-28`  
**Tên đề tài chính thức:** Xây dựng hệ thống phát hiện và sửa lỗi ngữ pháp tiếng Việt sử dụng mô hình ngôn ngữ  
**Phạm vi nghiên cứu cụ thể:** Sửa lỗi ngữ pháp, chính tả và chuẩn hóa phụ đề tiếng Việt từ nhận dạng giọng nói (ASR Post-Correction) sử dụng mô hình Sequence-to-Sequence  

---

## 1. Phát biểu bài toán (Problem Statement)
* **Đầu vào (Input):** Một câu phụ đề hoặc một tệp phụ đề định dạng `.srt` xuất phát từ hệ thống nhận dạng tiếng nói tự động (Speech-to-Text như Whisper), thường bị lỗi chính tả dấu thanh, mất dấu câu ngắt dòng, sai từ đồng âm hoặc đứt gãy ngữ pháp.
* **Đầu ra (Output):** Câu văn / tệp phụ đề `.srt` đã được khôi phục dấu câu, viết hoa đúng vị trí, sửa lỗi chính tả ngữ pháp, bảo toàn nguyên vẹn ý nghĩa ban đầu và mốc thời gian (timestamps).
* **Mục tiêu thực nghiệm & Giả thuyết:**
  * Kiểm chứng giả thuyết: *Mô hình ngôn ngữ chuỗi-sang-chuỗi (Seq2Seq Transformer như ViT5 / BARTpho) có khả năng tận dụng ngữ cảnh toàn câu để sửa các lỗi ASR tiếng Việt vượt trội hơn đáng kể so với phương pháp tra cứu từ điển / luật thông thường.*

---

## 2. Khảo sát và Chuẩn bị dữ liệu (Data)
* **Tập kiểm thử chuẩn (Gold Test Set):** Nhóm tự thu thập 300 - 500 đoạn subtitle thực tế xuất từ video tiếng Việt, sau đó tiến hành gán nhãn và sửa thủ công thành bản chuẩn (lưu định dạng JSONL gồm `noisy`, `clean`, `error_type`).
* **Tập huấn luyện (Training Set):** 
  * Nguồn câu chuẩn: Khoảng 30,000 - 50,000 câu tiếng Việt sạch từ báo chí / Wikipedia.
  * Phương pháp sinh nhiễu nhân tạo (Synthetic Noise Generator): Xóa dấu câu, viết thường toàn bộ, mô phỏng lỗi phát âm ngọng tiếng Việt (`l/n`, `d/gi/r`, `ch/tr`, `s/x`), lỗi gõ dấu thanh (`hỏi/ngã`), telex để tạo thành các cặp `(noisy, clean)`.

---

## 3. Tiền xử lý (Preprocessing)
* Chuẩn hóa Unicode dựng sẵn (NFC).
* Tách dòng subtitle, làm sạch các ký tự rác không thuộc bảng mã âm thanh.
* Sử dụng Tokenizer chính thức của mô hình tiền huấn luyện (SentencePiece với ViT5 hoặc Byte-Pair Encoding với BARTpho).
* Kiểm soát giới hạn độ dài câu tối đa (Max sequence length = 128 tokens).

---

## 4. Xây dựng Baseline đối chứng
* **Baseline 1 (Identity):** Giữ nguyên văn bản đầu ra từ ASR.
* **Baseline 2 (Rule-based & Dictionary):** Sử dụng danh sách từ điển tiếng Việt kết hợp thư viện khoảng cách xâu (Edit Distance / SymSpell) để sửa các từ không có trong từ điển sang từ có tần suất cao nhất gần nhất.

---

## 5. Phương pháp & Mô hình chính (Main Model)
* **Kiến trúc:** Sequence-to-Sequence (Encoder-Decoder Transformer).
* **Mô hình nền:** `VietAI/vit5-base` hoặc `vinai/bartpho-syllable`.
* **Kỹ thuật:** Fine-tuning bằng `Seq2SeqTrainer` của Hugging Face trên tập cặp câu tạo nhiễu, sử dụng hàm mất mát Cross-Entropy có nhãn mask padding.

---

## 6. Chỉ số đánh giá (Evaluation Metrics)
* **WER (Word Error Rate reduction):** Tỷ lệ giảm lỗi ở mức từ so với văn bản gốc từ ASR.
* **SacreBLEU / BLEU:** Đo lường mức độ trùng khớp ngữ nghĩa với câu chuẩn (Gold reference).
* **Edit-level Precision, Recall, F1:** Đo lường độ chính xác tại các vị trí xảy ra phép sửa (tránh sửa nhầm câu vốn đã đúng).

---

## 7. Phân tích lỗi (Error Analysis) & Demo
* **Error Analysis:** Lưu trữ và phân loại các mẫu mô hình sửa sai (over-correction, sửa đổi nghĩa của câu gốc, hallucination).
* **Demo:** Ứng dụng Web / Giao diện người dùng nhận vào file `.srt` thô, hiển thị so sánh song song giữa phụ đề trước và sau khi chuẩn hóa, cho phép tải về file `.srt` đã sửa hoàn chỉnh.
