"""Mô-đun sinh dữ liệu nhiễu nhân tạo (Synthetic Noise Generator)

Mô phỏng các lỗi đặc thù của hệ thống nhận dạng giọng nói tiếng Việt (ASR)
và lỗi gõ bàn phím:
1. Mất dấu câu (Punctuation removal)
2. Mất viết hoa (Lowercasing)
3. Nhầm lẫn âm đầu / phụ âm đồng âm (l/n, d/gi/r, ch/tr, s/x)
4. Nhầm lẫn dấu thanh (hỏi/ngã)
5. Mất dấu tiếng Việt (unidecode ngẫu nhiên)
"""

import random
import re
from typing import List, Tuple

# Cặp phụ âm thường bị nhầm lẫn trong phát âm tiếng Việt
CONSONANT_PAIRS = {
    "l": "n", "n": "l",
    "d": "gi", "gi": "d",
    "r": "d",
    "ch": "tr", "tr": "ch",
    "s": "x", "x": "s",
}

# Cặp nguyên âm có dấu hỏi - ngã thường bị lẫn lộn
TONE_PAIRS = {
    "ả": "ã", "ã": "ả",
    "ẻ": "ẽ", "ẽ": "ẻ",
    "ỉ": "ĩ", "ĩ": "ỉ",
    "ỏ": "õ", "õ": "ỏ",
    "ủ": "ũ", "ũ": "ủ",
    "ỷ": "ỹ", "ỹ": "ỷ",
    "ổ": "ỗ", "ỗ": "ổ",
    "ở": "ỡ", "ỡ": "ở",
    "ử": "ữ", "ữ": "ử",
    "ể": "ễ", "ễ": "ể",
}

# Bản đồ xóa dấu tiếng Việt
ACCENT_REMOVAL_MAP = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    "đ": "d",
}


class ASRNoiseGenerator:
    def __init__(
        self,
        drop_punct_prob: float = 0.85,
        lower_prob: float = 0.70,
        consonant_swap_prob: float = 0.15,
        tone_swap_prob: float = 0.15,
        drop_accent_word_prob: float = 0.10,
    ):
        self.drop_punct_prob = drop_punct_prob
        self.lower_prob = lower_prob
        self.consonant_swap_prob = consonant_swap_prob
        self.tone_swap_prob = tone_swap_prob
        self.drop_accent_word_prob = drop_accent_word_prob

    def remove_punctuation(self, text: str) -> str:
        """Xóa dấu câu mô phỏng đầu ra thô của Whisper/ASR."""
        return re.sub(r"[.,!?;:\"'()\[\]\-\/]", "", text)

    def remove_accents_from_word(self, word: str) -> str:
        """Xóa dấu toàn bộ từ tiếng Việt."""
        res = []
        for ch in word:
            lower_ch = ch.lower()
            if lower_ch in ACCENT_REMOVAL_MAP:
                rep = ACCENT_REMOVAL_MAP[lower_ch]
                res.append(rep.upper() if ch.isupper() else rep)
            else:
                res.append(ch)
        return "".join(res)

    def swap_tones(self, word: str) -> str:
        """Đổi dấu hỏi <-> ngã."""
        chars = list(word)
        for i, ch in enumerate(chars):
            lower_ch = ch.lower()
            if lower_ch in TONE_PAIRS and random.random() < self.tone_swap_prob:
                rep = TONE_PAIRS[lower_ch]
                chars[i] = rep.upper() if ch.isupper() else rep
        return "".join(chars)

    def swap_consonants(self, word: str) -> str:
        """Đổi phụ âm đầu mô phỏng nói ngọng/lẫn lộn âm."""
        lower_word = word.lower()
        for prefix, replacement in sorted(CONSONANT_PAIRS.items(), key=lambda x: -len(x[0])):
            if lower_word.startswith(prefix) and random.random() < self.consonant_swap_prob:
                is_upper = word[0].isupper()
                new_start = replacement.capitalize() if is_upper else replacement
                return new_start + word[len(prefix):]
        return word

    def apply_noise(self, clean_text: str) -> Tuple[str, List[str]]:
        """Nhận vào câu sạch và sinh ra câu nhiễu kèm nhãn các loại lỗi."""
        errors_applied = []
        words = clean_text.split()
        noisy_words = []

        # 1. Nhiễu ở mức từ
        for w in words:
            curr_w = w
            # Xác suất bỏ dấu nguyên từ
            if random.random() < self.drop_accent_word_prob:
                curr_w = self.remove_accents_from_word(curr_w)
                errors_applied.append("drop_accents")
            else:
                # Đổi phụ âm
                new_w = self.swap_consonants(curr_w)
                if new_w != curr_w:
                    errors_applied.append("consonant_swap")
                    curr_w = new_w
                # Đổi dấu thanh hỏi/ngã
                new_w = self.swap_tones(curr_w)
                if new_w != curr_w:
                    errors_applied.append("tone_swap")
                    curr_w = new_w

            noisy_words.append(curr_w)

        result_text = " ".join(noisy_words)

        # 2. Xóa dấu câu
        if random.random() < self.drop_punct_prob:
            result_text = self.remove_punctuation(result_text)
            errors_applied.append("drop_punctuation")

        # 3. Viết thường toàn bộ
        if random.random() < self.lower_prob:
            result_text = result_text.lower()
            errors_applied.append("lowercasing")

        # Chuẩn hóa khoảng trắng
        result_text = re.sub(r"\s+", " ", result_text).strip()

        return result_text, list(set(errors_applied))


if __name__ == "__main__":
    generator = ASRNoiseGenerator()
    samples = [
        "Tôi đi làm ở ngân hàng và dành tiền tiết kiệm.",
        "Bác sĩ khuyên bệnh nhân nên nghỉ ngơi, uống nhiều nước ấm.",
        "Chúng ta cần kiểm tra lại các tham số trong mô hình Seq2Seq này.",
    ]
    print("--- DEMO SINH DỮ LIỆU NHIỄU ASR ---")
    for s in samples:
        noisy, errs = generator.apply_noise(s)
        print(f"Clean : {s}")
        print(f"Noisy : {noisy}")
        print(f"Errors: {errs}\n")
