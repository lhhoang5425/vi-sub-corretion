# Vietnamese ASR Post-Correction & Subtitle Normalizer

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Model-BARTpho%20(bmd1905)-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/bmd1905/vietnamese-correction-v2)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lhhoang5425/vi-sub-corretion/pulls)

Hệ thống phát hiện, sửa lỗi ngữ pháp, chính tả và khôi phục dấu câu cho phụ đề tiếng Việt từ nhận dạng giọng nói (ASR Post-Correction) sử dụng mô hình Sequence-to-Sequence (`bmd1905/vietnamese-correction-v2`).

## Tổng quan

Đầu ra của các hệ thống nhận dạng tiếng nói tự động ASR (như OpenAI Whisper, Wav2Vec2) thường gặp các tật cố hữu:
- Sai từ đồng âm và dấu thanh tiếng Việt (d/gi/r, l/n, ch/tr, s/x, hỏi/ngã).
- Mất dấu câu và định dạng viết hoa đầu câu/danh từ riêng.
- Ngữ nghĩa bị đứt đoạn do câu nói tự nhiên.

Dự án này triển khai kiến trúc Sequence-to-Sequence (Seq2Seq Transformer) như một mô hình khử nhiễu (Denoising Model) để chuẩn hóa văn bản phụ đề mà vẫn bảo toàn chính xác 100% mốc thời gian (timestamps) của tệp `.srt`.

### Lộ trình 2 giai đoạn của đề tài
- **Giai đoạn 1 (GD1 - Trọng tâm hiện tại):** Huấn luyện và tinh chỉnh mô hình Seq2Seq trên văn bản câu đơn, xử lý triệt để khôi phục dấu thanh, viết hoa, dấu ngắt câu và sửa lỗi ngữ âm trên bộ dữ liệu 100k câu chuẩn hóa (0% rò rỉ train–val, tích hợp 10% mẫu sạch đối chứng) và đánh giá trên bộ Gold Test v2 (857 mẫu).
- **Giai đoạn 2 (GD2 - Kế hoạch mở rộng):** Tích hợp nối tiếp ngay sau đầu ra của Whisper ASR để tự động bóc tách và chuẩn hóa tệp phụ đề video thực tế `.srt` bảo toàn mốc thời gian.

## Kiến trúc hệ thống (Pipeline)

```text
[Video / Audio File]
         │
         ▼ (Whisper / ASR Engine)
[Raw SRT Subtitle] (lỗi dấu câu, sai chính tả, từ đồng âm, viết thường)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  1. SRT Parser: Bóc tách text & lưu vết Timestamps      │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  2. Pipeline Khử Nhiễu (Seq2Seq Denoising Model)       │
│     - Baseline 1: Identity (Giữ nguyên ASR)            │
│     - Baseline 2: Rule-based & SymSpell Dictionary     │
│     - Core Model: Fine-tuned bmd1905 (BARTpho)         │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  3. SRT Re-composer: Ghép text sạch vào Timestamps gốc │
└────────────────────────────────────────────────────────┘
         │
         ▼
[Clean & Polished SRT Subtitle] (Phụ đề mượt mà, đúng chuẩn)
```

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
git clone https://github.com/lhhoang5425/vi-sub-corretion.git
cd vi-sub-corretion

python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Sử dụng nhanh

### 1. Sửa lỗi câu đơn lẻ trong Python

```python
from src.models.inference import Seq2SeqCorrector

corrector = Seq2SeqCorrector(model_name_or_path="bmd1905/vietnamese-correction-v2")

raw_text = "toi di lam o ngan hang va gianh tien tiet kiem"
cleaned_text = corrector.predict(raw_text)

print(cleaned_text)
# Output: Tôi đi làm ở ngân hàng và dành tiền tiết kiệm.
```

### 2. Chuẩn hóa tệp phụ đề .SRT

```python
from src.srt_utils.srt_processor import SRTSubtitleProcessor
from src.models.inference import Seq2SeqCorrector

corrector = Seq2SeqCorrector(model_name_or_path="bmd1905/vietnamese-correction-v2")
processor = SRTSubtitleProcessor(corrector_model=corrector)

# Xử lý và xuất file mới giữ nguyên mốc thời gian
processor.process_srt_file("input.srt", "output_cleaned.srt")
```

### 3. Khởi chạy Web API (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Tài liệu API tương tác tại: `http://localhost:8000/docs`

- `POST /api/v1/correct-text`: Chuẩn hóa câu văn bản đơn.
- `POST /api/v1/correct-batch`: Chuẩn hóa danh sách câu theo batch.
- `POST /api/v1/correct-srt-string`: Nhận chuỗi SRT và trả về chuỗi SRT sạch.
- `POST /api/v1/correct-srt-file`: Tải lên tệp `.srt` và tải về tệp đã chuẩn hóa.

## Cấu trúc thư mục

```text
.
├── api/
│   └── main.py              # FastAPI server phục vụ tích hợp Web
├── data/
│   ├── raw/                 # Dữ liệu phụ đề/transcript gốc (.srt)
│   ├── synthetic/           # Dữ liệu sinh nhiễu nhân tạo cho huấn luyện
│   └── gold_test/           # Bộ kiểm thử chuẩn đa chiều (5 suite, 857 mẫu)
├── src/
│   ├── baseline/            # Mô hình đối chứng (Identity & Rule-based)
│   ├── data_processing/     # Sinh nhiễu ASR (asr_hard) và xây dựng bộ test tái lập
│   │   ├── noise_generator.py   # Bộ sinh nhiễu mô phỏng lỗi ngữ âm ASR
│   │   ├── corpus_builder.py    # Xây dựng cặp câu song song từ corpus
│   │   ├── dataset_loader.py    # Nạp và tiền xử lý dataset cho Trainer
│   │   ├── build_basic_test.py  # Script tái lập Gold Test GD1
│   │   ├── build_kichban_test.py# Script tái lập Gold Test GD2 từ SRT thật
│   │   └── build_corpus_test.py # Script tái lập test trên corpus văn học
│   ├── models/              # Huấn luyện và suy luận Seq2Seq
│   ├── evaluation/          # Đánh giá đa chiều (WER, CER, BLEU, Exact Match, Edit-F1)
│   └── srt_utils/           # Parser và ghép hoàn trả tệp SRT bảo toàn timestamps
├── notebooks/               # Jupyter/Colab notebooks huấn luyện (Lần 1 -> Lần 5)
├── tests/                   # Bộ kiểm thử đơn vị tự động (pytest)
├── DE_CUONG_NLP28.md        # Đề cương chi tiết học phần NLP-28
├── requirements.txt
└── README.md
```

## Kết quả Thực nghiệm Đột phá (Lần 5)

Đánh giá đối đầu trên tập kiểm thử chuẩn **Sample 25**:

| Phương pháp | BLEU ↑ | WER (%) ↓ | CER (%) ↓ | Exact Match (%) ↑ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Identity)** | 0.00 | 92.31% | 26.98% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Baseline 2 (Rule & Dict)** | 7.50 | 90.15% | 23.55% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Seq2Seq Lần 4 (100k legacy)** | 27.09 | 14.77% | — | — | 28.00% | 100.00% | 43.75% |
| **Seq2Seq Lần 5 (bmd1905 Full)** | **100.00** | **6.15%** | **2.57%** | **52.00%** | **52.00%** | **100.00%** | **68.42%** |

Chạy kiểm thử:
```bash
python -m pytest tests/
```

## Giấy phép

Dự án phát hành theo giấy phép MIT.
