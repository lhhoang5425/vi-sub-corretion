"""FastAPI Web Service cho Vietnamese ASR Post-Correction

Cung cấp REST API cho hệ thống Web của bạn để sửa lỗi text và tệp .srt.
Hỗ trợ CORS để gọi trực tiếp từ Frontend Web.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.models.inference import Seq2SeqCorrector
from src.srt_utils.srt_processor import SRTSubtitleProcessor

app = FastAPI(
    title="Vietnamese ASR Subtitle Post-Correction API",
    description="API phát hiện, sửa lỗi ngữ pháp & chính tả phụ đề tiếng Việt (NLP-28 Seq2Seq)",
    version="1.0.0",
)

# Cấu hình CORS để web của bạn gọi thoải mái
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo mô hình
MODEL_PATH = os.getenv("MODEL_NAME_OR_PATH", "bmd1905/vietnamese-correction-v2")
corrector = None
srt_processor = None


@app.on_event("startup")
def startup_event():
    global corrector, srt_processor
    print(f"[API Startup] Đang tải mô hình từ: {MODEL_PATH}")
    corrector = Seq2SeqCorrector(model_name_or_path=MODEL_PATH, use_fallback=True)
    srt_processor = SRTSubtitleProcessor(corrector_model=corrector)
    print("[API Startup] Dịch vụ đã sẵn sàng phục vụ!")


# Schemas
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


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": corrector.model is not None if corrector else False,
        "device": corrector.device if corrector else "unknown",
    }


@app.post("/api/v1/correct-text", response_model=TextCorrectionResponse)
def correct_single_text(req: TextCorrectionRequest):
    """Sửa lỗi một câu văn bản đơn."""
    if not req.text.strip():
        return TextCorrectionResponse(original=req.text, corrected="")
    corrected = corrector.predict(req.text)
    return TextCorrectionResponse(original=req.text, corrected=corrected)


@app.post("/api/v1/correct-batch", response_model=BatchCorrectionResponse)
def correct_batch_texts(req: BatchCorrectionRequest):
    """Sửa lỗi danh sách câu văn bản theo batch."""
    corrected_list = corrector.predict_batch(req.texts)
    results = [
        TextCorrectionResponse(original=orig, corrected=corr)
        for orig, corr in zip(req.texts, corrected_list)
    ]
    return BatchCorrectionResponse(results=results)


@app.post("/api/v1/correct-srt-string", response_class=PlainTextResponse)
def correct_srt_content(req: SRTStringRequest):
    """Nhận chuỗi nội dung .srt và trả về chuỗi .srt đã sửa chuẩn timestamps."""
    try:
        cleaned_srt = srt_processor.process_srt_string(req.srt_content)
        return cleaned_srt
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi phân tích cú pháp SRT: {str(e)}")


@app.post("/api/v1/correct-srt-file")
async def correct_srt_file_upload(file: UploadFile = File(...)):
    """Upload tệp .srt và nhận về tệp .srt đã được sửa lỗi."""
    if not file.filename.endswith(".srt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp định dạng .srt")

    content = await file.read()
    try:
        srt_text = content.decode("utf-8")
    except UnicodeDecodeError:
        srt_text = content.decode("utf-8-sig", errors="ignore")

    cleaned_srt = srt_processor.process_srt_string(srt_text)

    return PlainTextResponse(
        content=cleaned_srt,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=cleaned_{file.filename}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
