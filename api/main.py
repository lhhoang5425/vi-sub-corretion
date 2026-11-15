"""FastAPI Web Service cho Vietnamese ASR Post-Correction

Cung cấp REST API sửa lỗi văn bản / tệp .srt và phục vụ luôn giao diện Web
tại thư mục `web/` để giảng viên có thể demo trực tiếp trên trình duyệt.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models.inference import Seq2SeqCorrector
from src.srt_utils.srt_processor import SRTSubtitleProcessor

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

MODEL_PATH = os.getenv("MODEL_NAME_OR_PATH", "bmd1905/vietnamese-correction-v2")
USE_FALLBACK = os.getenv("USE_FALLBACK", "true").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[API Startup] Đang tải mô hình từ: {MODEL_PATH}")
    corrector = Seq2SeqCorrector(model_name_or_path=MODEL_PATH, use_fallback=USE_FALLBACK)
    app.state.corrector = corrector
    app.state.srt_processor = SRTSubtitleProcessor(corrector_model=corrector)

    if corrector.model is None:
        print("[API Startup] CẢNH BÁO: KHÔNG nạp được mô hình Seq2Seq, hệ thống đang chạy RULE BASELINE.")
        print("[API Startup] Kiểm tra MODEL_NAME_OR_PATH trong .env có trỏ tới checkpoint hợp lệ không.")
    else:
        print(f"[API Startup] Mô hình sẵn sàng trên thiết bị: {corrector.device}")

    yield

    app.state.corrector = None
    app.state.srt_processor = None


app = FastAPI(
    title="Vietnamese ASR Subtitle Post-Correction API",
    description="API phát hiện, sửa lỗi ngữ pháp & chính tả phụ đề tiếng Việt (NLP-28 Seq2Seq)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---------------------------------------------------------------


class TextCorrectionRequest(BaseModel):
    text: str = Field(..., example="toi di lam o ngan hang va gianh tien tiet kiem")


class TextCorrectionResponse(BaseModel):
    original: str
    corrected: str


class BatchCorrectionRequest(BaseModel):
    texts: List[str] = Field(..., example=["toi di lam", "hom nay rat vui"])


class BatchCorrectionResponse(BaseModel):
    results: List[TextCorrectionResponse]


class SRTStringRequest(BaseModel):
    srt_content: str = Field(..., description="Nội dung chuỗi của tệp .srt")


class SRTPairsResponse(BaseModel):
    count: int
    changed: int
    items: List[dict]


def _decode_upload(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("utf-8-sig", errors="ignore")


# --- Routes ----------------------------------------------------------------


@app.get("/health")
def health_check():
    """Kiểm tra trạng thái nạp mô hình trước khi demo."""
    corrector: Optional[Seq2SeqCorrector] = getattr(app.state, "corrector", None)
    if corrector is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được khởi tạo")

    seq2seq_ready = corrector.model is not None and corrector.tokenizer is not None
    return {
        "status": "healthy",
        "engine": "seq2seq" if seq2seq_ready else "rule-baseline",
        "model_name_or_path": corrector.model_name_or_path,
        "model_loaded": seq2seq_ready,
        "device": corrector.device,
    }


@app.post("/api/v1/correct-text", response_model=TextCorrectionResponse)
def correct_single_text(req: TextCorrectionRequest):
    """Sửa lỗi một câu văn bản đơn."""
    if not req.text.strip():
        return TextCorrectionResponse(original=req.text, corrected="")
    corrected = app.state.corrector.predict(req.text)
    return TextCorrectionResponse(original=req.text, corrected=corrected)


@app.post("/api/v1/correct-batch", response_model=BatchCorrectionResponse)
def correct_batch_texts(req: BatchCorrectionRequest):
    """Sửa lỗi danh sách câu văn bản theo batch."""
    corrected_list = app.state.corrector.predict_batch(req.texts)
    results = [
        TextCorrectionResponse(original=orig, corrected=corr)
        for orig, corr in zip(req.texts, corrected_list)
    ]
    return BatchCorrectionResponse(results=results)


@app.post("/api/v1/correct-srt-string", response_class=PlainTextResponse)
def correct_srt_content(req: SRTStringRequest):
    """Nhận chuỗi nội dung .srt và trả về chuỗi .srt đã sửa, giữ nguyên timestamps."""
    try:
        return app.state.srt_processor.process_srt_string(req.srt_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi phân tích cú pháp SRT: {e}")


@app.post("/api/v1/correct-srt-pairs", response_model=SRTPairsResponse)
def correct_srt_content_pairs(req: SRTStringRequest):
    """Trả về từng block phụ đề kèm câu gốc / câu đã sửa cho bảng đối chiếu song song."""
    try:
        items = app.state.srt_processor.process_srt_pairs(req.srt_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi phân tích cú pháp SRT: {e}")

    return SRTPairsResponse(
        count=len(items),
        changed=sum(1 for it in items if it["original"] != it["corrected"]),
        items=items,
    )


@app.post("/api/v1/correct-srt-file")
async def correct_srt_file_upload(file: UploadFile = File(...)):
    """Tải lên tệp .srt và nhận về tệp .srt đã được sửa lỗi."""
    if not file.filename.lower().endswith(".srt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp định dạng .srt")

    srt_text = _decode_upload(await file.read())
    cleaned_srt = app.state.srt_processor.process_srt_string(srt_text)

    out_name = f"cleaned_{Path(file.filename).name}"
    return PlainTextResponse(
        content=cleaned_srt,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.post("/api/v1/srt-to-txt", response_class=PlainTextResponse)
async def extract_txt_from_srt_endpoint(file: UploadFile = File(...)):
    """Trích xuất văn bản thuần túy (.txt) từ tệp phụ đề .srt (loại bỏ timestamps)."""
    if not file.filename.lower().endswith(".srt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp định dạng .srt")

    srt_text = _decode_upload(await file.read())
    pure_text = app.state.srt_processor.srt_to_txt(srt_text)

    out_name = f"{Path(file.filename).stem}.txt"
    return PlainTextResponse(
        content=pure_text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.post("/api/v1/clean-file", response_class=PlainTextResponse)
async def clean_file_endpoint(file: UploadFile = File(...), output_format: Optional[str] = "same"):
    """Chuẩn hóa tệp văn bản: hỗ trợ cả .srt và .txt.

    output_format: 'same' (giữ đuôi tệp gốc), 'txt' (xuất text thuần) hoặc 'srt'.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in (".srt", ".txt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp .srt hoặc .txt")

    raw_text = _decode_upload(await file.read())
    processor = app.state.srt_processor
    corrector = app.state.corrector

    if ext == ".srt":
        cleaned_srt = processor.process_srt_string(raw_text)
        if output_format == "txt":
            out_content = processor.srt_to_txt(cleaned_srt)
            out_name = f"cleaned_{Path(file.filename).stem}.txt"
        else:
            out_content = cleaned_srt
            out_name = f"cleaned_{Path(file.filename).name}"
    else:
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        cleaned_lines = corrector.predict_batch(lines) if lines else []
        out_content = "\n".join(cleaned_lines) + "\n"
        out_name = f"cleaned_{Path(file.filename).name}"

    return PlainTextResponse(
        content=out_content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
