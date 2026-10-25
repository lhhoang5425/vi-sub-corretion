# 🎙️ Vietnamese ASR Post-Correction & Subtitle Normalizer (Seq2Seq)
> **Đề tài NLP-28**: Xây dựng hệ thống phát hiện và sửa lỗi ngữ pháp / chính tả tiếng Việt sử dụng mô hình Sequence-to-Sequence cho phụ đề Speech-to-Text (ASR Post-Correction).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/Model-ViT5%20%7C%20BARTpho-orange)](https://huggingface.co/)

---

## 📌 1. Giới thiệu bài toán

Khi nhận dạng tiếng nói tự động (**ASR / Speech-to-Text** như Whisper, Wav2Vec2, Google STT) sang phụ đề tiếng Việt, đầu ra thường gặp các vấn đề lớn:
1. **Lỗi từ đồng âm & sai dấu thanh:** Phát âm lướt, ngọng địa phương (`d/gi/r`, `l/n`, `ch/tr`, `x/s`, hỏi/ngã) khiến câu văn sai lệch hoàn toàn ngữ nghĩa.
2. **Mất dấu câu & viết hoa:** Không có dấu chấm, phẩy ngắt câu, gây khó đọc cho phụ đề `.srt`.
3. **Từ đệm và ngữ cảnh nói:** Lặp từ, từ đệm nói tự nhiên (`ờ`, `thì`, `là`, `mà`) làm rối cấu trúc câu.

Dự án này triển khai kiến trúc **Sequence-to-Sequence (Seq2Seq)** đóng vai trò như một **bộ lọc khử nhiễu (Denoising Language Model)**: nhận vào câu phụ đề thô bị lỗi $\rightarrow$ phục hồi và viết lại thành câu văn chuẩn ngữ pháp, đúng chính tả, khôi phục dấu câu nhưng vẫn **bảo toàn 100% mốc thời gian (timestamps) của file `.srt`**.

---

## 🏗️ 2. Kiến trúc hệ thống (Pipeline Architecture)

```
[Video / Audio File]
         │
         ▼ (Whisper / ASR Engine)
[Raw SRT File] (Lỗi dấu câu, sai ngữ pháp, từ đồng âm)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  1. SRT Parser: Bóc tách text & lưu vết Timestamps      │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  2. Pipeline So sánh Thực nghiệm (NLP-28 Rubric)       │
│                                                        │
│  [Baseline]: SymSpell / Rule Dictionary / Regex        │
│  [Model Chính]: Fine-tuned ViT5-base / BARTpho-syllable│
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  3. SRT Re-composer: Ghép text sạch vào Timestamps gốc │
└────────────────────────────────────────────────────────┘
         │
         ▼
[Clean & Polished SRT File] (Phụ đề mượt mà, đúng chuẩn)
```

---

## 📁 3. Cấu trúc thư mục

```text
NLP_Project_seq2seq/
├── data/
│   ├── raw/                 # Dữ liệu phụ đề/transcript thô
│   ├── synthetic/           # Dữ liệu nhiễu nhân tạo (Noisy -> Clean)
│   └── gold_test/           # Bộ test chuẩn gán nhãn thủ công (300-500 mẫu)
├── src/
│   ├── baseline/            # Mô hình luật / từ điển đối chứng
│   │   └── rule_baseline.py
│   ├── data_processing/     # Sinh nhiễu mô phỏng ASR & nạp dataset
│   │   ├── noise_generator.py
│   │   └── dataset_loader.py
│   ├── models/              # Huấn luyện và dự đoán Seq2Seq
│   │   ├── train_seq2seq.py
│   │   └── inference.py
│   ├── evaluation/          # Đánh giá WER, BLEU, Edit-F1 theo Rubric
│   │   └── metrics.py
│   └── srt_utils/           # Parser và ghép file SRT
│       └── srt_processor.py
├── api/                     # Backend API phục vụ Web
│   └── main.py              # FastAPI Service
├── notebooks/               # Notebooks thực nghiệm trên Colab
├── tests/                   # Unit tests
├── DE_CUONG_NLP28.md        # Đề cương chi tiết nộp giảng viên
├── requirements.txt
└── README.md
```

---

## 🚀 4. Hướng dẫn cài đặt & Chạy nhanh

### 4.1. Khởi tạo môi trường
```bash
# Tạo virtual environment
python -m venv .venv
# Kích hoạt (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 4.2. Chạy thử nghiệm SRT Processor
```bash
python -m src.srt_utils.srt_processor
```

### 4.3. Chạy API Server (FastAPI) để kết nối vào Web
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập Swagger UI tại: `http://localhost:8000/docs`

---

## 📊 5. Tiêu chí đánh giá (Evaluation Metrics)

Theo đúng rubric môn học và bài toán GEC/ASR:
* **WER (Word Error Rate reduction):** Tỷ lệ giảm lỗi từ vựng so với câu gốc.
* **SacreBLEU / BLEU:** Đo độ tương đồng ngữ nghĩa so với câu tham chiếu chuẩn.
* **Edit-level Precision / Recall / F1:** Đo lường độ chính xác của các điểm sửa đổi (tránh sửa sai câu vốn đã đúng - over-correction).
