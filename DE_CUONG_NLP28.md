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

## 6. Chỉ số đánh giá & Kết quả Benchmark đối đầu (Evaluation & Benchmark)
* **WER (Word Error Rate):** Đo lường tỷ lệ từ bị sai khác so với câu chuẩn (càng thấp càng tốt).
* **SacreBLEU / BLEU:** Đo lường mức độ trùng khớp ngữ nghĩa n-gram với câu đáp án chuẩn (càng cao càng tốt).
* **Edit-level Precision, Recall, F1:** Đo lường độ chính xác và độ bao phủ tại các vị trí xảy ra phép sửa lỗi (tránh sửa nhầm câu vốn đã đúng).

### Bảng kết quả đánh giá thực nghiệm Benchmark định lượng (Tập Test vàng 25 mẫu)
Được đo trực tiếp giữa 3 phương pháp đối đầu trên cùng một tập dữ liệu kiểm thử thực tế:

| Phương pháp đối chứng | BLEU ↑ | WER (%) ↓ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ | Nhận xét tổng quan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1 (Identity)** | **0.00** | **92.31%** | 0.00% | 0.00% | **0.00%** | Giữ nguyên ASR lỗi, mất dấu tiếng Việt hoàn toàn |
| **Baseline 2 (Rule & Dictionary)** | **7.50** | **90.15%** | 0.00% | 0.00% | **0.00%** | Chỉ viết hoa đầu câu, bất lực trước câu mất dấu |
| **Seq2Seq (bmd1905 - 100,000 mẫu)** | **27.09** | **14.77%** | **28.00%** | **100.00%** | **43.75%** | **Vượt trội tuyệt đối: WER giảm sâu kỷ lục, Recall đạt 100%** |

---

## 7. Phân tích lỗi (Error Analysis) & Thử nghiệm thực tế (Demo)
* **Kết quả thử nghiệm ứng dụng thực tế:**
  * Mô hình xử lý thành công tệp phụ đề video thực tế (`kichban_2026-10-25.srt`), bảo toàn chính xác 100% các mốc thời gian (timestamps) và tự động xuất ra thư mục `output/` với tên phiên bản tự tăng (`kichban_cleaned_1.srt`).
  * Khôi phục dấu câu, viết hoa đầu câu và phục hồi dấu thanh tiếng Việt tự nhiên (độ chính xác từ đạt tới **85.23%**).
* **Phân tích lỗi tồn đọng & bài học rút ra:**
  1. **Khắc phục triệt để lỗi ảo giác chuỗi giải mã (Decoding Hallucination):** Tích hợp thành công bộ lọc Regex `clean_hallucinated_tail` và cơ chế Dynamic Token Bounding theo độ dài câu đầu vào, loại bỏ hoàn toàn các token lạ (tiếng Anh / ký tự ngoài bảng mã tiếng Việt) ở đuôi câu ngắn.
  2. **Hiện tượng sửa quá tay (Over-correction) do thiên lệch tần suất ngữ liệu (Language Model Prior Bias):** 
     * *Quan sát thực tế:* Cụm từ tiếng lóng / kênh YouTube *"chương trình tổ buôn 247"* bị mô hình tự ý sửa thành *"chương trình tổ chức 247"*.
     * *Nguyên nhân học thuật:* Trong dữ liệu huấn luyện, cụm *"chương trình tổ chức"* có tần suất xuất hiện áp đảo ($P \approx 0.95$), khiến giải thuật Beam Search ưu tiên chọn từ ngữ phổ biến thay vì tiếng lóng đời thường.
  3. **Lỗi cặp âm đồng âm địa phương ($d$ và $gi$):**
     * *Quan sát thực tế:* *"học làm giàu"* bị sửa nhầm thành *"học làm dầu"* do đầu vào ASR không dấu (`hoc lam dau`) và ngữ cảnh chưa đủ phân biệt giữa tài chính làm giàu với dầu mỏ/dầu nhớt.
* **Giải pháp khắc phục đề xuất:** Xây dựng danh sách từ điển ngoại lệ (Whitelist Dictionary) hoặc tích hợp module Nhận dạng thực thể tên riêng (NER) để cố định các danh từ riêng trước khi đưa qua mạng Seq2Seq.

---

## 8. Nhật ký thực nghiệm & Tiến triển qua các lần huấn luyện

Dưới đây là tiến trình nghiên cứu thực nghiệm được nhóm ghi nhận trực tiếp trên môi trường Google Colab (GPU NVIDIA Tesla T4 16GB VRAM):

### Bảng tổng hợp các đợt thực nghiệm qua các mốc dữ liệu

| Đợt chạy | Mô hình nền | Quy mô dữ liệu | Cấu hình (Epoch, Batch, LR) | Train Loss | Eval Loss | Grad Norm | Hiện tượng & Đánh giá |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lần 1** | `vinai/bartpho-syllable` | 133 câu | 3 Epochs, Batch 8, LR 5e-5 | 1.442 | 1.127 | 12.32 ⚠️ | **Underfitting (Chưa học đủ):** Mô hình thô chưa kịp học quy luật thêm dấu tiếng Việt do dữ liệu quá nhỏ so với 420M tham số. Gradient dao động mạnh. |
| **Lần 2A** | `bmd1905/vietnamese-correction-v2` | 133 câu | 3 Epochs, Batch 8, LR 5e-5 | 0.1956 | 0.2689 ⚠️ | 0.7434 ✅ | **Overfitting (Học vẹt) & Catastrophic Forgetting:** Eval loss tăng từ Epoch 1 (0.253) lên Epoch 2 (0.278). Mô hình bị ép học 133 câu nên sửa từ dị thường (`tiet kkiem`, `dau dan`). |
| **Lần 2B** | `bmd1905/vietnamese-correction-v2` | 133 câu | 2 Epochs, Batch 8, LR 3e-5 | 0.3540 | 0.3371 ✅ | 0.8120 ✅ | **Kiểm soát Overfitting:** Dừng ở 2 Epochs giúp Eval Loss giảm đều (0.359 $\rightarrow$ 0.337), mô hình giữ được độ tự nhiên hơn. |
| **Lần 3** | `bmd1905/vietnamese-correction-v2` | 10,000 câu (ShynBui + ASRNoiseGenerator) | 3 Epochs, Batch 16, LR 3e-5 | 0.0362 ✅ | 0.0184 ✅ | 0.25 - 1.7 ✅ | **Hội tụ tốt:** 9,000 train / 1,000 val (1,689 steps, 14 phút). Loss giảm sâu 98%, triệt tiêu học vẹt, bắt đầu sửa được phụ đề video thật. |
| **Lần 4 (MỚI NHẤT)** | `bmd1905/vietnamese-correction-v2` | **100,000 câu** (ShynBui + ASRNoiseGenerator) | **3 Epochs, Batch 16, LR 3e-5 (16,875 steps, 2h 12m)** | **0.03736** ✅ | **0.01307** ✅ | **0.04 - 1.4** ✅ | **Hội tụ đỉnh cao (Super Convergence):** Eval loss giảm sâu qua từng epoch (0.0276 $\rightarrow$ 0.0163 $\rightarrow$ 0.0130). WER giảm kỷ lục còn 14.77% (độ chính xác 85.23%), Edit-Recall đạt tuyệt đối 100.00%. |

### Đánh giá và bài học kinh nghiệm rút ra
1. **Quy luật về kích thước dữ liệu và hiệu năng:** Việc mở rộng tập dữ liệu từ 133 câu $\rightarrow$ 10,000 câu $\rightarrow$ 100,000 câu đã chứng minh rõ rệt: mô hình Seq2Seq Transformer chỉ thực sự phát huy sức mạnh khi được cung cấp đủ không gian ngữ cảnh. Giá trị `eval_loss` giảm sâu xuống **0.01307** và tỷ lệ lỗi từ `WER` giảm còn **14.77%** là minh chứng định lượng thuyết phục nhất.
2. **Khả năng bao phủ lỗi (Edit-Recall 100%):** Mô hình đạt mức bao phủ 100% đối với toàn bộ các câu ASR lỗi đầu vào, chủ động phục hồi dấu thanh và viết hoa mà không bỏ sót bất kỳ mẫu nào.
3. **Bài học về quản lý vòng đời mô hình (MLOps):** Xây dựng quy trình tự động hóa đánh số phiên bản Checkpoint (`v1, v2...`), tách bạch thư mục `input/` và `output/` giúp việc thử nghiệm trở nên khoa học, tránh lỗi đè file và tiết kiệm tài nguyên tính toán.

---

## 9. Nhật ký xử lý sự cố & Tiến trình khắc phục lỗi qua các giai đoạn (Troubleshooting & Bug Fix Log)

Nhằm minh bạch hóa quá trình nghiên cứu thực nghiệm và ghi nhận năng lực giải quyết vấn đề kỹ thuật (Engineering Problem Solving), nhóm tổng hợp toàn bộ các lỗi phát sinh qua từng giai đoạn và phương án khắc phục chi tiết:

### Bảng theo dõi lỗi phát sinh và phương án xử lý qua các đợt huấn luyện

| Giai đoạn / Mốc | Hiện tượng lỗi phát sinh (Symptom & Error) | Phân tích nguyên nhân kỹ thuật (Root Cause) | Phương án xử lý & Khắc phục cụ thể (Fix & Implementation) | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **Giai đoạn 1**<br>*(Khởi tạo & Lần 1)* | 1. `TypeError: 'dict' object is not an instance of Sequence` với ViT5 Tokenizer.<br>2. Xung đột tham số `evaluation_strategy` trong thư viện Transformers mới.<br>3. **Underfitting:** Loss = 1.127, Grad Norm vọt lên 12.32, không học được quy luật thêm dấu tiếng Việt. | 1. Output của tokenizer dạng Dict không tương thích cấu trúc dataset.<br>2. Thư viện `transformers >= 4.46` đổi tên thành `eval_strategy`.<br>3. Kích thước 133 câu quá nhỏ so với mạng BARTpho 420 triệu tham số, gradient dao động mất kiểm soát. | 1. Chuẩn hóa pipeline tokenization, fallback an toàn với `use_fast=False`.<br>2. Viết wrapper tự động nhận diện phiên bản `eval_strategy` vs `evaluation_strategy`.<br>3. Thay đổi chiến lược: Chuyển sang Transfer Learning 2 giai đoạn với backbone đã có nền tảng sửa lỗi tiếng Việt (`bmd1905`). | **Đã xử lý triệt để**<br>✅ |
| **Giai đoạn 2**<br>*(Lần 2A & 2B)* | **Overfitting (Học vẹt) & Quên thảm họa (Catastrophic Forgetting):** Eval Loss tăng ngược từ 0.253 lên 0.278 ở Epoch 2, mô hình sinh các từ biến dạng dị thường (`tiet kkiem`, `dau dan`). | Huấn luyện quá nhiều epoch (3 epochs) trên tập dữ liệu quá nhỏ (133 câu) khiến mạng nơ-ron bị ép ghi nhớ thuộc lòng, phá vỡ trọng số ngôn ngữ tự nhiên ban đầu. | 1. Ở lần 2B: Hạ Learning Rate xuống `3e-5` và áp dụng Early Stopping (dừng sớm ở 2 Epochs) để kìm hãm đà tăng loss.<br>2. Định hướng chiến lược dài hạn: Tự phát triển module `ASRNoiseGenerator` để giả lập nhiễu ASR trên quy mô lớn thay vì train trên tập dữ liệu nhỏ. | **Đã xử lý triệt để**<br>✅ |
| **Giai đoạn 3**<br>*(Lần 3 - 10k câu)* | 1. **Ảo giác giải mã (Decoding Hallucination / Token Leakage):** Mô hình sinh token tiếng Thái (`\u0e00-\u0e7f`) và tiếng Anh lặp vô tận ở đuôi câu ngắn.<br>2. Lỗi tràn ổ đĩa `C:\` do Hugging Face tự động tải cache mô hình.<br>3. Lỗi Colab mất checkpoint khi bị ngắt kết nối runtime (`FileNotFoundError`). | 1. Thuật toán Beam Search cố tìm token kết thúc (EOS) trong từ điển đa ngữ mBART nhưng bị kẹt ở các unigram xác suất thấp.<br>2. Cache mặc định nằm ở thư mục User trên ổ hệ thống Windows.<br>3. Ổ đĩa `/content/` của Colab là bộ nhớ tạm, tự động bị xóa sạch khi phiên bị thu hồi. | 1. Viết bộ lọc Regex `clean_hallucinated_tail()`, chặn trần độ dài động `dynamic_max_tokens = min(int(len * 1.3) + 12, max_len)` và thêm `repetition_penalty = 1.2`.<br>2. Tạo liên kết thư mục cứng `mklink /J` chuyển cache sang ổ `D:\Models\huggingface`.<br>3. Viết script nén `.zip` và tự động sao lưu checkpoint sang Google Drive ngay khi train xong. | **Đã xử lý triệt để**<br>✅ |
| **Giai đoạn 4**<br>*(Lần 4 - 100k câu)* | 1. **Lỗi vòng lặp file I/O:** Quét lặp file vô tận tạo ra các tệp `kichban_cleaned_cleaned_cleaned.srt`.<br>2. Lỗi nhầm lẫn tên file zip (`bartpho` vs `bmd1905`) khi nén lưu Drive.<br>3. **Hiện tượng Over-Correction:** Sửa sai tiếng lóng/tên riêng (*"tổ buôn 247"* $\rightarrow$ *"tổ chức 247"*) và cặp âm đồng âm địa phương ($d/gi$: *"làm giàu"* $\rightarrow$ *"làm dầu"*). | 1. Lệnh `glob("*.srt")` quét chung trong thư mục `input/` và đọc luôn các file đã sửa.<br>2. Code nén zip cũ chưa đồng bộ lại tên checkpoint sau khi đổi mô hình nền.<br>3. Mô hình bị thiên lệch tần suất ngôn ngữ (Language Model Prior Bias), ưu tiên cụm từ phổ biến trong sách báo chính luận hơn tiếng lóng đời thường. | 1. Tách riêng biệt thư mục `input/` và `output/` trên Google Drive, áp dụng hàm tự tăng phiên bản `_cleaned_1.srt`, `_cleaned_2.srt` và lọc `if "_cleaned" not in file.name`.<br>2. Đồng bộ hóa toàn bộ lệnh zip và sao lưu checkpoint tự động đánh số phiên bản `v1, v2, v3...`.<br>3. Đề xuất tích hợp module Từ điển ngoại lệ (Whitelist Dictionary) hoặc nhận dạng thực thể tên riêng (NER) để khóa danh từ riêng. | **Đã xử lý & Đề xuất nâng cao**<br>✅ / 🔄 |

---

## 10. Tổng quan các vấn đề tồn đọng sau đợt train Lần 4

> Đây là **bản tổng quan**. Phần phân tích chi tiết (số đo từng chỉ tiêu, cách kiểm chứng lại, và hướng triển khai đầy đủ cho Lần 5) được lưu riêng trong hồ sơ làm việc của nhóm.

### 10.1. Tóm tắt

Mô hình Lần 4 vẫn tốt, và hai con số công bố ở Mục 6 — WER 14.77%, BLEU 27.09 — **vẫn hợp lệ** (đã kiểm: tập kiểm thử vàng không rò rỉ vào tập train). Tuy nhiên khi rà soát lại khâu tạo dữ liệu và khâu đo, nhóm phát hiện một loại vấn đề khác với các lỗi đã ghi ở Mục 9: những khiếm khuyết **không gây lỗi chương trình**. Mô hình vẫn train xong, loss vẫn giảm đẹp, benchmark vẫn ra số — nhưng **ý nghĩa của các con số đó bị lệch**.

Ba vấn đề nghiêm trọng nhất:

1. **Tập validation rò rỉ 20.8% vào tập train**, do dataset nguồn trùng lặp nội bộ ~21% mà code lại chia tập theo vị trí dòng thay vì theo nội dung. Hệ quả: `eval_loss = 0.01307` — bằng chứng chính cho kết luận "hội tụ đỉnh cao" ở Mục 8 — được đo trên tập có ~1/5 là dữ liệu mô hình đã thuộc lòng. Hướng giảm đơn điệu qua 3 epoch vẫn là bằng chứng chống overfitting, nhưng **độ lớn tuyệt đối của các giá trị không còn đáng tin**.
2. **Phân bố lỗi trong nhiễu huấn luyện không khớp tập kiểm thử.** Khảo sát 1.500 câu cho thấy tập huấn luyện **giữ nguyên 100% dấu thanh** (bộ nhiễu chỉ bỏ dấu ở 10% số từ, và `lowercasing` không đụng tới dấu thanh), trong khi tập kiểm thử yêu cầu khôi phục dấu cho **25/25 câu mất trắng dấu**. Phần fine-tune của nhóm vì vậy dạy mô hình thêm dấu câu và viết hoa; khả năng khôi phục dấu thanh nhiều khả năng đến từ backbone `bmd1905` có sẵn.
3. **Tập kiểm thử vàng chỉ 25 mẫu, một loại lỗi, và không có câu sạch.** Mỗi câu chiếm 4% chỉ số nên phương sai lấn át mọi cải thiện cần đo. Nghiêm trọng hơn: vì không có câu sạch, nhóm "câu vốn đã đúng nhưng bị sửa nhầm" **luôn rỗng** — nghĩa là hiện tượng over-correction mô tả ở Mục 7 (khóa nêu ví dụ *"tổ buôn 247"* thành *"tổ chức 247"*) **chưa từng được đo**, mới chỉ được tường thuật.

**Điểm chung:** cả ba đều nằm ở khâu **dữ liệu** và khâu **đo**, không nằm ở kiến trúc mô hình. Sửa được mà không phải đổi mô hình, và phần lớn chi phí là **0 phút GPU**.

### 10.2. Bảng tổng hợp các vấn đề

| Mã | Vấn đề | Mức độ | Ảnh hưởng trực tiếp | Trạng thái |
| :---: | :--- | :---: | :--- | :---: |
| **P1** | Validation rò rỉ 20.8% vào train (dataset trùng lặp nội bộ 21%) | 🔴 Nghiêm trọng | `eval_loss` 0.01307 — Mục 8 | ⚠️ Chưa xử lý |
| **P2** | Train giữ 90% dấu thanh, test yêu cầu khôi phục từ câu mất 100% dấu | 🔴 Nghiêm trọng | Ý nghĩa WER 14.77% / BLEU 27.09 — Mục 6 | 🔄 Đã sửa code |
| **P3** | Tập vàng 25 mẫu, một loại lỗi, không có câu sạch | 🔴 Nghiêm trọng | Độ tin cậy Mục 6; Mục 7 chưa được đo | ⚠️ Chưa xử lý |
| **P4** | 79% token là padding thừa; `DataCollatorForSeq2Seq` bị vô hiệu hóa | 🟡 Trung bình | Hiệu quả & thời gian train — Mục 8 | ✅ Đã sửa code |
| **P5** | Không có cặp `(sạch → sạch)` nào trong tập huấn luyện | 🟡 Trung bình | Edit-Precision 28% — Mục 6 & 7 | 🔄 Đã sửa code |
| **P6** | Tham số giải mã chồng hai hình phạt lặp, chưa tinh chỉnh lại | 🟢 Chưa xác định | Nguy cơ sửa thừa — Mục 7 | 🔄 Đã có công cụ đo |
| **P7** | `reports/benchmark_summary.md` trong repo chứa số sai, mâu thuẫn Mục 6 | 🟡 Trung bình | Người đọc repo thấy hai bộ số đối chọi | ⚠️ Chưa xử lý |

*🔴 làm sai lệch kết luận nghiên cứu · 🟡 làm giảm hiệu năng hoặc gây mâu thuẫn hồ sơ · 🟢 chưa chắc gây hại nhưng chưa được kiểm chứng.*

Về **P7**: tệp `reports/benchmark_summary.md` đang ghi Seq2Seq đạt BLEU 0.00 / WER 92.00%, mâu thuẫn với Mục 6. Số 92.00% thực chất là kết quả đo **mô hình gốc chưa fine-tune** trong một lần chạy local (do biến môi trường `MODEL_NAME_OR_PATH` đặt ở cấp hệ thống đè lên tệp `.env`), gần như trùng với Baseline 1 Identity (92.31%). Con số này **có giá trị riêng**: nó chứng minh mô hình gốc hầu như không khôi phục được dấu thanh, nên bước fine-tune là cần thiết chứ không phải tùy chọn — nhưng phải được dán nhãn đúng nguồn gốc, không được đặt cạnh số của Lần 4 mà không giải thích.

### 10.3. Hướng khắc phục và thứ tự ưu tiên

| Bậc | Việc | Chi phí | Lý do đứng ở bậc này |
| :---: | :--- | :--- | :--- |
| **1** | P6 — tìm kiếm tham số giải mã trên tập validation | ~5 phút GPU, **không train lại** | Rẻ nhất; đo được ngay, và chốt một tham số mà mọi phép đo sau đều phụ thuộc |
| **2** | P3 — xây tập kiểm thử v2: 4 lát cắt, ~200 mẫu, có lát câu sạch | 0 giờ code + ~1–2 giờ gõ nhãn | **Phải làm trước khi train**: thước đo định nghĩa "tốt" nghĩa là gì |
| **3** | P1 — chia tập theo hash nội dung + khử trùng lặp + lọc câu rác | 0 (sửa code sinh dữ liệu) | Chặn tái diễn ở mọi đợt sau |
| **4** | P2 + P5 — dùng hồ sơ nhiễu `asr_hard` (bỏ dấu cả câu + 10% câu sạch) | 0 (đã xong code) | Sửa nguyên nhân gốc của cả hai vấn đề |
| **5** | P4 — chuyển sang pad động theo batch | 0 (đã xong code) | Tăng hiệu quả train; đã có cờ A/B để tách bạch đóng góp |
| **6** | Train Lần 5 và benchmark lại **cả Lần 4** trên cùng thước mới | ~1–2 giờ T4 | Chỉ có ý nghĩa sau khi bậc 1–5 xong |

**Nguyên tắc xuyên suốt:** cùng một thước đo, cùng một cấu hình giải mã, đo nhiều mô hình — thì khoảng chênh lệch mới là thật. Hiện mỗi đợt dùng thước khác nhau nên **chưa thể kết luận hơn thua**.

**Cảnh báo khi đọc kết quả:** nếu WER trên tập vàng v2 **xấu hơn** 14.77%, đó **không** phải dấu hiệu mô hình tệ đi. Con số 14.77% được đo trên 25 câu khá thuận lợi (tất cả đều mất dấu — đúng loại dữ liệu mô hình được train để xử lý). Thước mới đa dạng và khắt khe hơn, nên số xấu hơn là **dấu hiệu thước tốt**.

### 10.4. Bài học phương pháp luận

1. **Thước đo phải đáng tin trước khi tối ưu mô hình.** Bốn đợt train tập trung vào loss và quy mô dữ liệu, trong khi tập kiểm thử — thứ định nghĩa thành công — dừng ở 25 mẫu tự viết.
2. **Loss thấp không đồng nghĩa với học đúng.** `eval_loss = 0.01307` thấp một phần vì tác vụ dễ (P2), một phần vì tập đo trùng tập học (P1) — cả hai đều không liên quan chất lượng mô hình.
3. **Dữ liệu huấn luyện phải khớp phân bố lỗi của triển khai, không chỉ khớp về số lượng.** Tăng 10k lên 100k cải thiện loss rõ rệt, nhưng nếu lệch phân bố thì 100k đó vẫn không dạy được kỹ năng trọng tâm.
4. **Phải có dữ liệu đối chứng âm.** Không có câu sạch trong tập đo thì không bao giờ phát hiện được mô hình phá câu đúng — P3 và P5 cùng tạo ra điểm mù này.
5. **Dataset công khai không mặc nhiên đáng tin.** Nguồn dữ liệu dùng cho Lần 4 trùng lặp 21% và chứa văn bản mô tả-AI lặp lại 60 lần. Cần kiểm toán dữ liệu trước khi dùng, không chỉ nạp rồi chạy.
