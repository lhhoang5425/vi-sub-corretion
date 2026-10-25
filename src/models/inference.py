"""Mô-đun dự đoán / suy luận Seq2Seq (Inference Engine)

Hỗ trợ suy luận cả câu đơn và batch câu với mô hình ViT5 hoặc BARTpho.
Tự động fallback về baseline nếu chưa tải được checkpoint nặng hoặc chạy trên máy CPU.
"""

import os
from typing import List, Optional

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

from ..baseline.rule_baseline import RuleAndDictionaryBaseline


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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
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

        # Chạy suy luận qua Seq2Seq model
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )

        clean_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_text.strip()

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
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

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                )

            batch_results = [
                self.tokenizer.decode(out, skip_special_tokens=True).strip()
                for out in outputs
            ]
            results.extend(batch_results)

        return results
