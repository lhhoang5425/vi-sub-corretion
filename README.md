# Vietnamese ASR Post-Correction & Subtitle Normalizer

Hệ thống phát hiện, sửa lỗi ngữ pháp, chính tả và khôi phục dấu câu cho phụ đề tiếng Việt từ nhận dạng giọng nói (ASR) sử dụng mô hình Sequence-to-Sequence (ViT5 / BARTpho).

## Tổng quan

Đầu ra của các hệ thống nhận dạng tiếng nói tự động (Whisper, Google STT, Wav2Vec2) thường gặp các vấn đề:
- Sai từ đồng âm và dấu thanh (d/gi/r, l/n, ch/tr, s/x, hỏi/ngã).
- Mất dấu câu và định dạng viết hoa.
- Ngữ nghĩa bị đứt đoạn do câu nói tự nhiên.

Dự án này triển khai kiến trúc Sequence-to-Sequence (Seq2Seq) như một mô hình khử nhiễu (Denoising Model) để chuẩn hóa văn bản phụ đề mà vẫn bảo toàn chính xác mốc thời gian (timestamps) của tệp `.srt`.

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

## Cấu trúc dự án

```text
.
├── api/
│   └── main.py              # FastAPI server phục vụ tích hợp Web
├── data/
│   ├── raw/                 # Dữ liệu phụ đề/transcript gốc
│   ├── synthetic/           # Dữ liệu sinh nhiễu nhân tạo cho huấn luyện
│   └── gold_test/           # Bộ kiểm thử chuẩn gán nhãn thủ công
├── src/
│   ├── baseline/            # Mô hình đối chứng (Rule & Dictionary)
│   ├── data_processing/     # Sinh nhiễu mô phỏng ASR và nạp dữ liệu
│   ├── models/              # Huấn luyện và suy luận Seq2Seq
│   ├── evaluation/          # Đánh giá WER, BLEU, Edit-F1
│   └── srt_utils/           # Parser và ghép tệp SRT
├── notebooks/               # Notebook thực nghiệm
├── tests/                   # Kiểm thử đơn vị
├── DE_CUONG_NLP28.md        # Đề cương đề tài
├── requirements.txt
└── README.md
```

## Đánh giá hiệu năng

Hệ thống được đánh giá trên tập dữ liệu kiểm thử độc lập thông qua 3 chỉ số chính:
- **WER (Word Error Rate):** Đánh giá mức độ giảm tỷ lệ lỗi từ vựng.
- **SacreBLEU:** Đo lường độ tương đồng ngữ nghĩa so với câu tham chiếu chuẩn.
- **Edit-level Precision / Recall / F1:** Đo lường độ chính xác tại các vị trí chỉnh sửa (hạn chế hiện tượng over-correction).

Chạy kiểm thử đánh giá:
```bash
python -m pytest tests/
```

## Giấy phép

Dự án phát hành theo giấy phép MIT.
