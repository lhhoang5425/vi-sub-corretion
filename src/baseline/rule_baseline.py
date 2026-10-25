"""Mô-đun mô hình đối chứng (Baseline Models)

Theo đúng yêu cầu của đề tài NLP-28:
- Baseline 1: Identity Baseline (giữ nguyên câu đầu vào từ ASR)
- Baseline 2: Rule & Dictionary Baseline (áp dụng luật chính tả, từ điển tra cứu
  và khôi phục viết hoa/dấu câu đơn giản)
"""

import re
from typing import List, Optional


class IdentityBaseline:
    """Baseline giữ nguyên văn bản gốc (Zero-change)."""

    def predict(self, text: str) -> str:
        return text

    def predict_batch(self, texts: List[str], **kwargs) -> List[str]:
        return texts


class RuleAndDictionaryBaseline:
    """Baseline dựa trên luật (Rule-based) và từ điển tra cứu các cặp từ hay sai."""

    COMMON_CORRECTIONS = {
        # Lỗi chính tả & nhầm lẫn thông dụng
        "xúc tích": "súc tích",
        "sáng lạn": "xán lạn",
        "dành tiền": "dành tiền",
        "giành tiền": "dành tiền",
        "gianh tien": "dành tiền",
        "danh tien": "dành tiền",
        "cọ sát": "cọ xát",
        "thăm quan": "tham quan",
        "chuẩn hoá": "chuẩn hóa",
        "giã từ": "dã từ",
        "nghỉ ngơi": "nghỉ ngơi",
    }

    def __init__(self, custom_dict: Optional[dict] = None):
        self.dict = dict(self.COMMON_CORRECTIONS)
        if custom_dict:
            self.dict.update(custom_dict)

    def restore_capitalization_and_period(self, text: str) -> str:
        """Khôi phục viết hoa chữ cái đầu và thêm dấu chấm cuối câu."""
        text = text.strip()
        if not text:
            return ""

        # Viết hoa chữ cái đầu câu
        text = text[0].upper() + text[1:]

        # Thêm dấu chấm cuối nếu chưa có dấu kết thúc
        if text[-1] not in [".", "!", "?"]:
            text += "."
        return text

    def predict(self, text: str) -> str:
        """Xử lý sửa lỗi cho 1 câu."""
        res = text
        # Thay thế theo từ điển
        for wrong, right in self.dict.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            res = pattern.sub(right, res)

        # Khôi phục viết hoa và chấm câu
        res = self.restore_capitalization_and_period(res)
        return res

    def predict_batch(self, texts: List[str], **kwargs) -> List[str]:
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    baseline = RuleAndDictionaryBaseline()
    sample = "toi di lam o ngan hang va gianh tien tiet kiem"
    print("Input   :", sample)
    print("Baseline:", baseline.predict(sample))
