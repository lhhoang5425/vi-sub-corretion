"""Mô-đun dự đoán / suy luận Seq2Seq (Inference Engine)

Hỗ trợ suy luận cả câu đơn và batch câu với mô hình ViT5 hoặc BARTpho.
Tự động fallback về baseline nếu chưa tải được checkpoint nặng hoặc chạy trên máy CPU.
"""

import os
import re
from typing import List, Optional

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

from ..baseline.rule_baseline import RuleAndDictionaryBaseline


def clean_hallucinated_tail(text: str, input_text: str = "") -> str:
    """Lọc bỏ ảo giác giải mã (Decoding Hallucination / Token Leakage).

    1. Loại bỏ các ký tự ngoài bảng mã tiếng Việt/Latin (tiếng Thái, Trung, Ả Rập, Cyrillic...).
    2. Ràng buộc độ dài: Nếu câu sinh ra dài vượt mức bất thường so với câu gốc (> 1.8 lần),
       cắt bỏ phần đuôi bị sinh lố sau dấu kết thúc câu (. ! ?).
    3. Dọn sạch dấu câu lơ lửng ở cuối (dấu phẩy, hai chấm, gạch nối...).
    """
    if not text:
        return ""

    # 1. Loại bỏ ký tự tiếng Thái (\u0e00-\u0e7f), CJK (\u4e00-\u9fff), Cyrillic (\u0400-\u04ff), Arabic (\u0600-\u06ff)
    cleaned = re.sub(r"[\u0e00-\u0e7f\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff\u3040-\u30ff]", "", text)

    # 2. Chuẩn hoá khoảng trắng
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 3. Kiểm soát độ dài so với câu gốc
    if input_text:
        in_words = input_text.strip().split()
        out_words = cleaned.split()
        if len(in_words) > 0 and len(out_words) > int(len(in_words) * 1.8) + 5:
            cutoff_len = int(len(in_words) * 1.4) + 3
            truncated = " ".join(out_words[:cutoff_len])
            m = re.search(r"[\.\?!][^\.\?!]*$", truncated)
            if m:
                cleaned = truncated[: m.end()].strip()
            else:
                cleaned = truncated.strip()

    # 4. Xóa dấu câu lơ lửng ở cuối
    cleaned = re.sub(r"[\s,;:\-]+$", "", cleaned).strip()
    return cleaned


class Seq2SeqCorrector:
    """Class wrapper phục vụ suy luận mô hình Seq2Seq sửa lỗi văn bản."""

    def __init__(
        self,
        model_name_or_path: str = "bmd1905/vietnamese-correction-v2",
        device: Optional[str] = None,
        max_length: int = 128,
        use_fallback: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.use_fallback = use_fallback
        self.fallback_model = RuleAndDictionaryBaseline()

        if device is None:
            self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if torch is None or AutoModelForSeq2SeqLM is None:
            if self.use_fallback:
                print("[Seq2SeqCorrector] Chưa cài đặt 'torch' hoặc 'transformers'. Tự động chạy chế độ Baseline.")
                return
            else:
                raise ImportError("Vui lòng cài đặt thư viện: pip install torch transformers")

        try:
            print(f"[Seq2SeqCorrector] Đang tải mô hình: {self.model_name_or_path} trên {self.device}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
            print("[Seq2SeqCorrector] Khởi tạo mô hình thành công!")
        except Exception as e:
            print(f"[Seq2SeqCorrector] CẢNH BÁO: Không thể nạp mô hình ({e}).")
            if self.use_fallback:
                print("[Seq2SeqCorrector] Tự động chuyển sang chế độ Rule/Dictionary Baseline.")
                self.model = None
                self.tokenizer = None
            else:
                raise e

    def predict(self, text: str) -> str:
        """Sửa lỗi cho 1 câu đơn."""
        if not text or not text.strip():
            return ""

        if self.model is None or self.tokenizer is None:
            return self.fallback_model.predict(text)

        # Chạy suy luận qua Seq2Seq model với ràng buộc độ dài động (chống ảo giác đuôi)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        input_num_tokens = inputs["input_ids"].shape[1]
        dynamic_max_tokens = min(max(int(input_num_tokens * 1.3) + 12, 20), self.max_length)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=dynamic_max_tokens,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )

        clean_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_hallucinated_tail(clean_text, input_text=text)

    def predict_batch(self, texts: List[str], batch_size: int = 8, **kwargs) -> List[str]:
        """Sửa lỗi cho danh sách câu theo batch (tối ưu tốc độ cho tệp phụ đề .srt)."""
        if not texts:
            return []

        if self.model is None or self.tokenizer is None:
            return self.fallback_model.predict_batch(texts)

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            max_input_len = inputs["input_ids"].shape[1]
            dynamic_max_tokens = min(max(int(max_input_len * 1.3) + 12, 20), self.max_length)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=dynamic_max_tokens,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                )

            batch_results = [
                clean_hallucinated_tail(self.tokenizer.decode(out, skip_special_tokens=True), input_text=src)
                for out, src in zip(outputs, batch_texts)
            ]
            results.extend(batch_results)

        return results
