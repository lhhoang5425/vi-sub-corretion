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
* **Error Analysis thực tế:**
  * **Thành công:** Mô hình khôi phục dấu câu, viết hoa đầu câu, sửa lỗi ngữ pháp tiếng Việt rất tự nhiên (ví dụ: *"Xin chào tất cả quý vị và các bạn, cùng đến với chương trình..."*).
  * **Lỗi tồn đọng phát hiện:** Hiện tượng **ảo giác chuỗi giải mã (Decoding Hallucination / Token Leakage)** ở đuôi câu ngắn khi Beam Search cố gắng sinh thêm từ tiếp theo và bị vướng vào các token đa ngôn ngữ trong từ điển mBART (xuất hiện một vài từ tiếng Anh / ký tự lạ ở cuối câu).
  * **Giải pháp khắc phục tiếp theo:** Bổ sung bước hậu xử lý (Post-processing) lọc ký tự ngoài bảng mã tiếng Việt và ràng buộc tỉ lệ độ dài `max_new_tokens` dựa theo độ dài câu gốc.
* **Demo:** Ứng dụng xử lý tệp `.srt` thực tế (đã kiểm thử thành công trên tệp phụ đề `kichban_2026-10-25.srt`), bảo toàn chính xác 100% các mốc thời gian (timestamps) và xuất ra tệp `_cleaned.srt` hoàn chỉnh.

---

## 8. Nhật ký thực nghiệm & Tiến triển qua các lần huấn luyện

Dưới đây là tiến trình nghiên cứu thực nghiệm được nhóm ghi nhận trực tiếp trên môi trường Google Colab (GPU NVIDIA Tesla T4 16GB VRAM):

### Bảng tổng hợp các đợt thực nghiệm

| Đợt chạy | Mô hình nền | Quy mô dữ liệu | Cấu hình (Epoch, Batch, LR) | Train Loss | Eval Loss | Grad Norm | Hiện tượng & Đánh giá |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lần 1** | `vinai/bartpho-syllable` | 133 câu | 3 Epochs, Batch 8, LR 5e-5 | 1.442 | 1.127 | 12.32 ⚠️ | **Underfitting (Chưa học đủ):** Mô hình thô chưa kịp học quy luật thêm dấu tiếng Việt do dữ liệu quá nhỏ so với 420M tham số. Gradient dao động mạnh. |
| **Lần 2A** | `bmd1905/vietnamese-correction-v2` | 133 câu | 3 Epochs, Batch 8, LR 5e-5 | 0.1956 | 0.2689 ⚠️ | 0.7434 ✅ | **Overfitting (Học vẹt) & Catastrophic Forgetting:** Eval loss tăng từ Epoch 1 (0.253) lên Epoch 2 (0.278). Mô hình bị ép học 133 câu nên sửa từ dị thường (`tiet kkiem`, `dau dan`). |
| **Lần 2B** | `bmd1905/vietnamese-correction-v2` | 133 câu | 2 Epochs, Batch 8, LR 3e-5 | 0.3540 | 0.3371 ✅ | 0.8120 ✅ | **Kiểm soát Overfitting:** Dừng ở 2 Epochs giúp Eval Loss giảm đều (0.359 $\rightarrow$ 0.337), mô hình giữ được độ tự nhiên hơn. |
| **Lần 3 (Mới nhất)** | `bmd1905/vietnamese-correction-v2` | **10,000 câu** (ShynBui + ASRNoiseGenerator) | **3 Epochs, Batch 16, LR 3e-5** | **0.0362** ✅ | **0.0184** ✅ | **0.25 - 1.7** ✅ | **Hội tụ tối ưu (Optimal Convergence):** 9,000 câu train / 1,000 câu val (1,689 steps, 14 phút). Loss giảm sâu 98%, triệt tiêu hoàn toàn học vẹt. Sửa được phụ đề video thực tế. |

### Đánh giá và bài học kinh nghiệm rút ra
1. **Quy luật về kích thước dữ liệu:** Đối với mô hình Sequence-to-Sequence hàng trăm triệu tham số, tập dữ liệu dưới 1,000 câu không đủ để mô hình tổng quát hóa. Việc mở rộng lên 10,000 câu kết hợp bộ tạo nhiễu `ASRNoiseGenerator` đã đưa giá trị hàm mất mát kiểm thử (`eval_loss`) giảm ngoạn mục xuống **0.0184**.
2. **Chiến lược Transfer Learning 2 giai đoạn:** Khởi đầu từ một mô hình đã có năng lực sửa lỗi tiếng Việt cơ bản (`bmd1905`) giúp tiết kiệm hàng trăm giờ huấn luyện so với việc đi từ mô hình nền thô.
3. **Kế hoạch hoàn thiện tiếp theo:**
   * Tích hợp bộ lọc Regex làm sạch các token ngoại lai ở đuôi câu nhằm giải quyết triệt để lỗi ảo giác.
   * Chạy đánh giá định lượng bằng mã nguồn `src/evaluation/metrics.py` để xuất bảng số liệu BLEU, WER và Edit-F1 hoàn chỉnh.
