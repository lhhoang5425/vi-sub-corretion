# ĐỀ CƯƠNG ĐỀ TÀI BÀI TẬP NHÓM NLP

**Học phần:** Xử lý ngôn ngữ tự nhiên (NLP) — UNETI 2026  
**Giảng viên hướng dẫn:** Thầy Bùi Văn Tân  
**Mã đề tài:** `NLP-28` (Nhóm D: Các bài toán ứng dụng NLP — Mức độ: Chuyên sâu)  
**Tên đề tài chính thức:** Xây dựng hệ thống phát hiện và sửa lỗi ngữ pháp tiếng Việt sử dụng mô hình ngôn ngữ  
**Phạm vi nghiên cứu cụ thể:** Sửa lỗi ngữ pháp, chính tả và chuẩn hóa phụ đề tiếng Việt từ nhận dạng giọng nói (ASR Post-Correction) sử dụng mô hình Sequence-to-Sequence  

---

## 1. Giới thiệu & Phát biểu bài toán (Problem Statement)
* **Bối cảnh & Động lực:** Đầu ra của các hệ thống nhận dạng tiếng nói tự động tiếng Việt (như Whisper, Wav2Vec2) thường xuyên gặp các lỗi: mất dấu câu, viết thường toàn bộ, sai dấu thanh do phát âm, sai từ đồng âm địa phương (`ch/tr`, `s/x`, `d/gi/r`, `l/n`) hoặc đứt gãy ngữ nghĩa.
* **Đầu vào (Input):** Câu văn bản tiếng Việt hoặc tệp phụ đề định dạng `.srt` xuất phát từ hệ thống ASR (có thể có lỗi hoặc câu vốn đã đúng ngữ pháp).
* **Đầu ra (Output):** Phát hiện loại lỗi, vị trí sửa và câu văn / tệp phụ đề `.srt` đã được khôi phục dấu câu, viết hoa, sửa lỗi chính tả ngữ pháp, bảo toàn nguyên vẹn ý nghĩa ban đầu và mốc thời gian (timestamps).
* **Mục tiêu thực nghiệm & Giả thuyết nghiên cứu:**
  * Xây dựng pipeline hoàn chỉnh từ dữ liệu, huấn luyện đến suy luận và đánh giá với schema rõ ràng.
  * Kiểm chứng giả thuyết: *Mô hình ngôn ngữ Sequence-to-Sequence (Seq2Seq Transformer) có khả năng tận dụng ngữ cảnh toàn câu để sửa lỗi ASR vượt trội hơn hẳn các phương pháp tra cứu từ điển / luật thông thường.*
  * Báo cáo đầy đủ chỉ số định lượng, chi phí thực hiện, phân tích hiện tượng over-correction (sửa nhầm câu đúng) và các trường hợp biên.

---

## 2. Khảo sát Dữ liệu, Schema & Phân chia tập (Dataset & Split)

### 2.1. Schema dữ liệu gán nhãn
Mỗi mẫu dữ liệu trong hệ thống tuân thủ định dạng chuẩn JSONL:
```json
{
  "id": "sample_001",
  "noisy": "toi di lam o ngan hang va gianh tien tiet kiem",
  "clean": "Tôi đi làm ở ngân hàng và dành tiền tiết kiệm.",
  "error_type": "tones, punctuation, capitalization, homophone",
  "source": "ShynBui_filtered"
}
```

### 2.2. Nguồn dữ liệu & Quy trình chuẩn bị
* **Tập huấn luyện (Training Set):** 
  * Nguồn dữ liệu: 220,000 dòng từ corpus tiếng Việt (`ShynBui/Vietnamese_spelling_error`), qua bộ lọc rác, loại bỏ câu ký tự lạ và khử trùng lặp nội dung còn 100,000 câu sạch chuẩn.
  * **Cơ chế phân chia chống rò rỉ (Data Leakage Prevention):** Sử dụng mã băm SHA-256 trên nội dung câu sạch trước khi sinh nhiễu để chia tập (tỷ lệ 90% train — 90,028 mẫu / 10% val — 9,972 mẫu, seed 42). **Triệt tiêu hoàn toàn tỷ lệ trùng lặp train–val về 0.00%** (khắc phục triệt để lỗi rò rỉ 20.8% ở các phiên bản trước).
  * **Hồ sơ sinh nhiễu nhân tạo (`ASRNoiseGenerator` - `asr_hard`):** Mô phỏng toàn diện cơ chế lỗi nhận dạng của mô hình ASR tiếng Việt:
    * Xóa 100% dấu thanh và chuyển chữ thường toàn câu.
    * Sinh lỗi ngữ âm địa phương (`l/n`, `ch/tr`, `s/x`, `d/gi/r`, hỏi/ngã).
    * **Tích hợp 10% mẫu đối chứng âm sạch (`identity` - `noisy == clean`):** Dạy mô hình biết giữ nguyên khi câu đầu vào vốn đã đúng, ngăn chặn hiện tượng tự ý sửa đổi từ ngữ (over-correction).

### 2.3. Bộ kiểm thử chuẩn đa chiều (Gold Test Set v2 — 857 mẫu)
Để đánh giá toàn diện năng lực mô hình thay vì phụ thuộc vào một tập kiểm thử nhỏ, nhóm xây dựng 5 bộ test độc lập:
1. `sample25` (25 mẫu): Tập chuẩn đối đầu trực tiếp qua từng giai đoạn nghiên cứu.
2. `unaccent` (200 mẫu): Lát cắt mất 100% dấu thanh và dấu câu (đo năng lực phục hồi cơ bản).
3. `asrhard` (200 mẫu): Lát cắt nhiễu tổng hợp gồm lỗi ngọng địa phương, telex và 15 câu sạch đối chứng.
4. `multisource` (300 mẫu): Kết hợp 150 câu ShynBui và 150 câu Wikipedia đa lĩnh vực (đo khả năng tổng quát hóa miền).
5. `kichban` (132 mẫu): Tập phụ đề thực tế trích xuất từ video YouTube gán nhãn thủ công (gồm 72 câu giữ nguyên và 19 câu độ tin cậy thấp).

---

## 3. Tiền xử lý (Preprocessing) & Tối ưu hóa tính toán
* **Chuẩn hóa văn bản:** Chuẩn hóa Unicode dựng sẵn (NFC), làm sạch khoảng trắng thừa và ký tự không thuộc bảng mã âm thanh.
* **Đóng gói dữ liệu động (Dynamic Batching):** Sử dụng `DataCollatorForSeq2Seq` của Hugging Face để tự động đệm padding động theo độ dài câu dài nhất trong batch thay vì cố định chiều dài 128 tokens cho toàn bộ mẫu. Giải pháp này giúp loại bỏ 79% token đệm thừa, tiết kiệm bộ nhớ VRAM và rút ngắn đáng kể thời gian huấn luyện.
* **Giới hạn độ dài:** Kiểm soát độ dài đầu vào/đầu ra (`max_source_length = 128`, `max_target_length = 128` tokens).

---

## 4. Thiết kế Thực nghiệm & Các phương pháp đối chứng

### 4.1. Baseline 1: Giữ nguyên (Identity Baseline)
* **Mô tả:** Giữ nguyên 100% văn bản đầu ra từ mô hình nhận dạng giọng nói ASR mà không can thiệp bất kỳ thao tác sửa đổi nào.
* **Ý nghĩa học thuật:** Đây là đối chứng bắt buộc để thiết lập mức sàn hiệu năng và kiểm tra xem hệ thống có làm giảm chất lượng câu hay không.

### 4.2. Baseline 2: Luật kết hợp từ điển tần suất (Rule-based & Dictionary Baseline)
* **Mô tả:** Sử dụng bảng từ điển từ vựng tiếng Việt kết hợp thuật toán khoảng cách xâu (Edit Distance / SymSpell) để sửa các từ không có trong từ điển sang từ có tần suất cao nhất gần nhất, kết hợp quy tắc viết hoa ký tự đầu câu và sau dấu chấm.
* **Hạn chế:** Không có khả năng hiểu ngữ cảnh toàn câu để khôi phục dấu thanh cho các từ đồng âm hoặc câu mất dấu hoàn toàn.

### 4.3. Phương pháp chính: Mô hình Sequence-to-Sequence (Seq2Seq Transformer)
* **Kiến trúc & Backbone:** Tinh chỉnh (Fine-tuning) trên mô hình `bmd1905/vietnamese-correction-v2` (kiến trúc Encoder-Decoder Transformer phát triển trên nền tảng BARTpho).
* **Kỹ thuật tối ưu hóa Lần 5:**
  * Huấn luyện với hàm mất mát Cross-Entropy tích hợp **Label Smoothing** (`label_smoothing_factor = 0.1`) nhằm làm mềm phân bố xác suất token mục tiêu, hạn chế mô hình quá tự tin vào từ ngữ phổ biến.
  * Tinh chỉnh không gian giải mã suy luận (Inference Decoding): `num_beams = 5`, `repetition_penalty = 1.2`, `no_repeat_ngram_size = 3`.
  * Bộ lọc hậu xử lý Regex `clean_hallucinated_tail` và Dynamic Token Bounding để triệt tiêu hiện tượng ảo giác sinh token lạ ở đuôi câu ngắn.

---

## 5. Chỉ số đánh giá & Kết quả Thực nghiệm (Evaluation & Benchmark)

### 5.1. Hệ thống chỉ số đánh giá
* **Chỉ số mức từ & ký tự:**
  * **WER (Word Error Rate) ↓:** Tỷ lệ từ sai khác so với đáp án chuẩn (càng thấp càng tốt).
  * **CER (Character Error Rate) ↓:** Tỷ lệ ký tự sai khác (nhạy bén phát hiện lỗi sai lệch dấu thanh).
* **Chỉ số mức câu & n-gram:**
  * **BLEU / ROUGE-L ↑:** Đo lường mức độ trùng khớp n-gram và độ bao phủ ngữ nghĩa so với câu chuẩn.
  * **Exact Match (EM) ↑:** Tỷ lệ câu trùng khớp chính xác 100% với đáp án chuẩn của con người.
* **Chỉ số mức can thiệp sửa đổi (Edit-level Metrics):**
  * **Edit-Precision (Edit-P) ↑:** Tỷ lệ câu sửa đúng trong số các câu mô hình có can thiệp sửa (đo lường trực tiếp hiện tượng over-correction).
  * **Edit-Recall (Edit-R) ↑:** Tỷ lệ bao phủ các câu lỗi cần được can thiệp sửa.
  * **Edit-F1 ↑:** Trung bình điều hòa giữa Precision và Recall mức sửa đổi.

### 5.2. Bảng kết quả đánh giá đối đầu trên tập Gold Test chuẩn (Sample 25)

| Phương pháp đối chứng | BLEU ↑ | WER (%) ↓ | CER (%) ↓ | Exact Match (%) ↑ | ROUGE-L (%) ↑ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1 (Identity)** | **0.00** | **92.31%** | 26.98% | **0.00%** | 7.58% | 0.00% | 0.00% | **0.00%** | Giữ nguyên ASR lỗi, mất dấu tiếng Việt hoàn toàn |
| **Baseline 2 (Rule & Dictionary)** | **7.50** | **90.15%** | 23.55% | **0.00%** | 9.72% | 0.00% | 0.00% | **0.00%** | Chỉ viết hoa đầu câu, bất lực trước câu mất dấu |
| **Seq2Seq Lần 4 (100k legacy)** | **27.09** | **14.77%** | — | — | — | 28.00% | 100.00% | **43.75%** | Giảm lỗi tốt nhưng bị rò rỉ val, over-correct |
| **Seq2Seq Lần 5 (Tối ưu Full - MỚI NHẤT)** | **100.00** | **6.15%** | **2.57%** | **52.00%** | **93.89%** | **52.00%** | **100.00%** | **68.42%** | **Vượt trội toàn diện: WER giảm sâu kỷ lục còn 6.15%, Edit-F1 tăng vọt lên 68.42%** |

### 5.3. Bảng kết quả Benchmark toàn diện của Mô hình Lần 5 trên 5 lát cắt kiểm thử

| Bộ dữ liệu kiểm thử (Suite) | Số mẫu | BLEU ↑ | WER (%) ↓ | CER (%) ↓ | Exact Match (%) ↑ | ROUGE-L (%) ↑ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Sample Test (sample25)** | 25 | **100.00** | **6.15%** | 2.57% | **52.00%** | 93.89% | 52.00% | 100.00% | **68.42%** |
| **2. Mất dấu hoàn toàn (unaccent)** | 200 | **100.00** | **7.93%** | 2.45% | **35.50%** | 91.89% | 35.50% | 100.00% | **52.40%** |
| **3. Nhiễu ASR tổng hợp (asrhard)** | 200 | **100.00** | **7.01%** | 2.24% | **41.00%** | 92.98% | 37.89% | 100.00% | **54.96%** |
| **4. Đa nguồn ShynBui & Wiki (multisource)** | 300 | **100.00** | **16.20%** | 6.28% | **20.67%** | 83.74% | 20.67% | 100.00% | **34.25%** |
| **5. Phụ đề ASR video thực tế (kichban)** | 132 | **79.66** | **9.61%** | 5.57% | **34.85%** | 90.69% | 3.45% | 6.25% | **4.44%** |

---

## 6. Phân tích Kết quả, Hiện tượng Lỗi & Thử nghiệm Demo

### 6.1. Đánh giá chất lượng thực tế & Thử nghiệm Demo (.SRT Processor)
* Hệ thống đã thử nghiệm xử lý trực tiếp tệp phụ đề thực tế `kichban_2026-10-25.srt` (132 đoạn phụ đề xuất từ Whisper ASR).
* **Kết quả:** Mô hình tự động bóc tách text, khôi phục dấu câu, viết hoa và thêm dấu tiếng Việt tự nhiên, sau đó ghép nối chính xác 100% mốc thời gian (timestamps) gốc để xuất tệp phụ đề mới. Tỷ lệ lỗi từ WER trên tập thực tế chỉ ở mức **9.61%**, CER chỉ **5.57%**.

### 6.2. Phân tích các trường hợp khó & lỗi tồn đọng
1. **Nhiều cách sửa đúng (Multiple Valid Corrections):** Trong các câu văn giao tiếp đời thường, một câu đầu vào có thể có nhiều cách sửa dấu câu hoặc từ đồng nghĩa tương đương nhau. Việc đối chiếu với một đáp án duy nhất khiến chỉ số Exact Match chưa phản ánh hết những câu mô hình sửa đúng về mặt ngữ nghĩa nhưng khác về cách ngắt dấu câu.
2. **Hiện tượng sửa nhầm do thiên lệch ngữ liệu (Over-correction & Prior Bias):** 
   * Đối với từ lóng hoặc tên riêng hiếm gặp (ví dụ: kênh YouTube *"tổ buôn 247"*), mô hình có xu hướng sửa thành từ có xác suất cao trong sách báo (*"tổ chức 247"*).
   * Kỹ thuật thêm 10% mẫu sạch và `label_smoothing` ở Lần 5 đã cải thiện Edit-Precision từ 28% lên **52.00%**, tuy nhiên vẫn cần cơ chế Whitelist Dictionary cho danh từ riêng chuyên biệt.
3. **Câu mơ hồ ngữ cảnh (Context Ambiguity):** Với các câu quá ngắn hoặc mất dấu hoàn toàn tạo ra nhiều nghĩa phân vân (`hoc lam dau` $\rightarrow$ *"học làm giàu"* hay *"học làm dầu"*), mô hình bắt buộc phải dựa vào xác suất tiên nghiệm của từ đứng cạnh.

---

## 7. Chi phí Thực hiện, Tài nguyên & Khả năng Tái lập (Reproducibility)
* **Tài nguyên tính toán:** Quá trình huấn luyện thực hiện hoàn toàn trên môi trường Google Colab tiêu chuẩn sử dụng GPU NVIDIA Tesla T4 (16GB VRAM), RAM hệ thống 12GB.
* **Thời gian huấn luyện:** ~2 giờ cho 3 Epochs trên tập dữ liệu 90,028 mẫu (batch size 16, gradient accumulation steps 2).
* **Khả năng tái lập:** Toàn bộ quy trình từ tải dữ liệu, sinh nhiễu theo seed 42 cố định, huấn luyện và benchmark được đóng gói hoàn chỉnh trong notebook `seq2seq_train_5_optimize.ipynb` và các script Python module hóa trong thư mục `src/`.

---

## 8. Nhật ký thực nghiệm & Tiến triển qua các lần huấn luyện

Dưới đây là tiến trình nghiên cứu thực nghiệm được nhóm ghi nhận trực tiếp qua các giai đoạn cải tiến:

### Bảng tổng hợp các đợt thực nghiệm qua các mốc dữ liệu

| Đợt chạy | Mô hình nền | Quy mô dữ liệu | Cấu hình (Epoch, Batch, LR) | Train Loss | Eval Loss | Grad Norm | Hiện tượng & Đánh giá |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lần 1** | `vinai/bartpho-syllable` | 133 câu | 3 Epochs, Batch 8, LR 5e-5 | 1.442 | 1.127 | 12.32 ⚠️ | **Underfitting (Chưa học đủ):** Mô hình thô chưa kịp học quy luật thêm dấu tiếng Việt do dữ liệu quá nhỏ so với 420M tham số. Gradient dao động mạnh. |
| **Lần 2A** | `bmd1905/vietnamese-correction-v2` | 133 câu | 3 Epochs, Batch 8, LR 5e-5 | 0.1956 | 0.2689 ⚠️ | 0.7434 ✅ | **Overfitting (Học vẹt) & Catastrophic Forgetting:** Eval loss tăng từ Epoch 1 (0.253) lên Epoch 2 (0.278). Mô hình bị ép học 133 câu nên sửa từ dị thường (`tiet kkiem`, `dau dan`). |
| **Lần 2B** | `bmd1905/vietnamese-correction-v2` | 133 câu | 2 Epochs, Batch 8, LR 3e-5 | 0.3540 | 0.3371 ✅ | 0.8120 ✅ | **Kiểm soát Overfitting:** Dừng ở 2 Epochs giúp Eval Loss giảm đều (0.359 $\rightarrow$ 0.337), mô hình giữ được độ tự nhiên hơn. |
| **Lần 3** | `bmd1905/vietnamese-correction-v2` | 10,000 câu (ShynBui + ASRNoiseGenerator) | 3 Epochs, Batch 16, LR 3e-5 | 0.0362 ✅ | 0.0184 ✅ | 0.25 - 1.7 ✅ | **Hội tụ tốt:** 9,000 train / 1,000 val (1,689 steps, 14 phút). Loss giảm sâu 98%, triệt tiêu học vẹt, bắt đầu sửa được phụ đề video thật. |
| **Lần 4** | `bmd1905/vietnamese-correction-v2` | 100,000 câu (ShynBui legacy) | 3 Epochs, Batch 16, LR 3e-5 (16,875 steps, 2h 12m) | 0.03736 ✅ | 0.01307 ⚠️ *(rò rỉ)* | 0.04 - 1.4 ✅ | **Hội tụ cao nhưng lệch thước đo:** Eval loss giảm đẹp nhưng bị rò rỉ 20.8% dữ liệu giữa train và val; bộ nhiễu chưa xóa dấu triệt để; over-correction cao (Edit-P 28%). |
| **Lần 5 (HOÀN THIỆN)** | `bmd1905/vietnamese-correction-v2` | **100,000 câu chuẩn hóa** (90,028 Train / 9,972 Val - `asr_hard`) | **3 Epochs, Batch 16, LR 3e-5, Label Smoothing 0.1, Dynamic Padding** | **0.03512** ✅ | **0.01241** ✅ *(chuẩn)* | **0.05 - 1.2** ✅ | **Hội tụ chuẩn xác & Đột phá chất lượng:** Triệt tiêu hoàn toàn rò rỉ (0% overlap). WER giảm sâu kỷ lục còn 6.15%, Edit-P đạt 52.00%, Edit-F1 đạt 68.42% trên tập chuẩn. |

---

## 9. Kết luận & Hướng phát triển

1. **Kết luận chung:** 
   * Đề tài đã xây dựng thành công pipeline hoàn chỉnh chuẩn hóa phụ đề tiếng Việt từ nhận dạng giọng nói ASR bằng mô hình Sequence-to-Sequence.
   * Kết quả thực nghiệm đối đầu trên các tập kiểm thử đã chứng minh tính đúng đắn của giả thuyết: mô hình Seq2Seq tối ưu Lần 5 (`bmd1905-sub-corrector_lan5_asr_hard_full_ls1`) đạt hiệu năng áp đảo hai phương pháp Baseline (WER giảm sâu từ **92.31%** xuống **6.15%**, CER chỉ **2.57%**, Edit-Recall đạt **100.00%** và Exact Match đạt **52.00%**).
2. **Hướng phát triển:**
   * Tích hợp thêm module Nhận dạng thực thể tên riêng (NER) nhẹ hoặc danh sách Whitelist Dictionary nhằm bảo toàn nguyên vẹn danh từ riêng và tiếng lóng.
   * Mở rộng ngữ cảnh nhiều câu (Multi-sentence Context) trong phụ đề `.srt` để cải thiện khả năng phân biệt ngữ nghĩa các từ đồng âm phức tạp.
   * Đóng gói mô hình thành dịch vụ API microservice (FastAPI kết hợp ONNX Runtime / TensorRT-LLM) phục vụ chuẩn hóa phụ đề thời gian thực trên các nền tảng video.
