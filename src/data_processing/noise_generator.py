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

# ── Đổi dấu thay vì bỏ dấu ────────────────────────────────────────────────────
# Lỗi phổ biến nhất của Whisper tiếng Việt KHÔNG phải "mất dấu" mà là "sai dấu".
# Đo trên `data/raw/kichban_2026-10-25.srt` (phụ đề Whisper thật): 99,3% câu CÒN
# dấu thanh, nhưng sai ở nhiều chỗ — "làm giàu" -> "làm dầu", "củng cố" -> "cùng cố",
# "rực rỡ" -> "dực rỡ", "nồng nhiệt" -> "nông nhiệt", "dạy" -> "dậy".
# Hai bảng dưới tái tạo đúng loại lỗi đó.

# Nhóm nguyên âm cùng gốc, khác DẤU THANH
_TONE_GROUPS = [
    "aàáảãạ", "ăằắẳẵặ", "âầấẩẫậ",
    "eèéẻẽẹ", "êềếểễệ",
    "iìíỉĩị",
    "oòóỏõọ", "ôồốổỗộ", "ơờớởỡợ",
    "uùúủũụ", "ưừứửữự",
    "yỳýỷỹỵ",
]

# Nhóm nguyên âm cùng DẤU, khác ÂM (a/ă/â, o/ô/ơ, u/ư, e/ê)
_VOWEL_GROUPS = ["aăâ", "oôơ", "uư", "eê"]


def _build_shift_map(groups):
    """Bảng tra: ký tự -> các ký tự có thể thay thế (khác chính nó)."""
    out = {}
    for g in groups:
        for ch in g:
            out[ch] = [c for c in g if c != ch]
    return out


TONE_SHIFT_MAP = _build_shift_map(_TONE_GROUPS)
VOWEL_SHIFT_MAP = _build_shift_map(_VOWEL_GROUPS)

# Tập ký tự ĐANG mang dấu thanh (khác dấu móc/mũ). Dùng để biết vị trí nào được phép
# đổi dấu — nếu cho phép dán thêm dấu vào nguyên âm trơn, từ sẽ có hai dấu thanh
# ("báo" -> "báõ"), thứ không tồn tại trong tiếng Việt.
_TONED_CHARS = set("àáảãạ ằắẳẵặ ầấẩẫậ èéẻẽẹ ềếểễệ ìíỉĩị òóỏõọ ồốổỗộ ờớởỡợ ùúủũụ ừứửữự ỳýỷỹỵ".replace(" ", ""))


# Các bộ thông số nhiễu dựng sẵn.
#
# "legacy" là nguyên bản hành vi cũ (đo trên 1.500 câu: giữ lại ~90% dấu thanh,
# 0% câu mất trắng dấu). Vấn đề là tập kiểm thử vàng của đề tài lại có 25/25 câu
# MẤT TOÀN BỘ dấu thanh, nên mô hình train bằng "legacy" không bao giờ gặp đúng
# loại đầu vào nó sẽ phải xử lý. "asr_hard" cân lại cho khớp phân bố đó.
NOISE_PROFILES = {
    "legacy": {
        "drop_punct_prob": 0.85,
        "lower_prob": 0.70,
        "consonant_swap_prob": 0.15,
        "tone_swap_prob": 0.15,
        "drop_accent_word_prob": 0.10,
        "full_unaccent_prob": 0.00,
        "identity_prob": 0.00,
    },
    "asr_hard": {
        "drop_punct_prob": 1.00,
        "lower_prob": 1.00,
        "consonant_swap_prob": 0.25,
        "tone_swap_prob": 0.25,
        "drop_accent_word_prob": 0.35,
        # 0.10 câu được giữ sạch TRƯỚC, nên nhánh này chỉ nhận ~90% còn lại:
        # 0.85 * 0.90 ≈ 76.5% câu mất trắng dấu thanh (đo thực tế, xem log verify).
        # Gold test là 100%, nhưng ép nhiễu giống hệt 25 câu test sẽ thành overfit
        # phân bố lỗi; chừa ~13% nhiễu nhẹ để mô hình vẫn xử lý được câu ASR có dấu.
        "full_unaccent_prob": 0.85,
        "identity_prob": 0.10,
    },
    # Khớp CHÍNH XÁC dạng của `data/gold_test/sample_test.jsonl` (25 câu): 100% câu
    # mất trắng dấu thanh, bỏ hết dấu câu, viết thường toàn bộ. Dùng để dựng tập test
    # mở rộng cho BÀI TOÁN GỐC mà đề cương đã tuyên bố.
    "unaccent_all": {
        "drop_punct_prob": 1.00,
        "lower_prob": 1.00,
        "consonant_swap_prob": 0.00,
        "tone_swap_prob": 0.00,
        "drop_accent_word_prob": 0.00,
        "full_unaccent_prob": 1.00,
        "identity_prob": 0.00,
    },
    # Mô phỏng ĐÚNG lỗi của Whisper tiếng Việt, đo từ `data/raw/kichban_2026-10-25.srt`:
    # giữ dấu thanh (99,3% câu), giữ dấu câu (81,1%), viết hoa (82,4%). Lỗi chính là
    # SAI DẤU và nhầm phụ âm đầu — KHÔNG phải mất dấu. Vì vậy `full_unaccent_prob = 0`
    # và `drop_accent_word_prob = 0`, thay vào đó là đổi dấu (tone_shift / vowel_shift).
    #
    # Các xác suất dưới đây chọn để khớp số đo trên file thật: file có 45,5% block bị
    # Whisper làm sai (mỗi câu vài chỗ trên ~16 từ), và 54,5% block đúng hoàn toàn.
    # `identity_prob` cao (0,45) chính là để tái tạo nhóm "đúng hoàn toàn" đó — nhóm mà
    # `legacy` sinh ra 0% và `asr_hard` chỉ 10%.
    "whisper_like": {
        "drop_punct_prob": 0.20,
        "lower_prob": 0.18,
        "consonant_swap_prob": 0.08,
        "tone_swap_prob": 0.05,
        "drop_accent_word_prob": 0.00,
        "full_unaccent_prob": 0.00,
        "identity_prob": 0.45,
        "tone_shift_prob": 0.12,
        "vowel_shift_prob": 0.06,
    },
}


class ASRNoiseGenerator:
    def __init__(
        self,
        drop_punct_prob: float = 0.85,
        lower_prob: float = 0.70,
        consonant_swap_prob: float = 0.15,
        tone_swap_prob: float = 0.15,
        drop_accent_word_prob: float = 0.10,
        full_unaccent_prob: float = 0.0,
        identity_prob: float = 0.0,
        tone_shift_prob: float = 0.0,
        vowel_shift_prob: float = 0.0,
        profile: str = None,
        seed: int = None,
    ):
        if profile is not None:
            if profile not in NOISE_PROFILES:
                raise ValueError(f"Profile nhiễu không tồn tại: {profile}. "
                                 f"Hợp lệ: {', '.join(NOISE_PROFILES)}")
            cfg = NOISE_PROFILES[profile]
            drop_punct_prob = cfg["drop_punct_prob"]
            lower_prob = cfg["lower_prob"]
            consonant_swap_prob = cfg["consonant_swap_prob"]
            tone_swap_prob = cfg["tone_swap_prob"]
            drop_accent_word_prob = cfg["drop_accent_word_prob"]
            full_unaccent_prob = cfg["full_unaccent_prob"]
            identity_prob = cfg["identity_prob"]
            # Hồ sơ cũ (legacy/asr_hard/unaccent_all) không có hai khoá này -> mặc định 0
            tone_shift_prob = cfg.get("tone_shift_prob", 0.0)
            vowel_shift_prob = cfg.get("vowel_shift_prob", 0.0)

        self.drop_punct_prob = drop_punct_prob
        self.lower_prob = lower_prob
        self.consonant_swap_prob = consonant_swap_prob
        self.tone_swap_prob = tone_swap_prob
        self.drop_accent_word_prob = drop_accent_word_prob
        self.full_unaccent_prob = full_unaccent_prob
        self.identity_prob = identity_prob
        self.tone_shift_prob = tone_shift_prob
        self.vowel_shift_prob = vowel_shift_prob

        # Bộ sinh số ngẫu nhiên RIÊNG cho từng thực thể khi có seed.
        # Không seed -> dùng module `random` toàn cục như trước (hành vi cũ, không tái lập).
        # Có seed -> mỗi lần chạy sinh ra đúng cùng một tập dữ liệu nhiễu, điều kiện bắt buộc
        # để tái lập thực nghiệm: cùng code + cùng seed phải ra cùng tập train.
        self.rng = random.Random(seed) if seed is not None else random

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
            if lower_ch in TONE_PAIRS and self.rng.random() < self.tone_swap_prob:
                rep = TONE_PAIRS[lower_ch]
                chars[i] = rep.upper() if ch.isupper() else rep
        return "".join(chars)

    def swap_consonants(self, word: str) -> str:
        """Đổi phụ âm đầu mô phỏng nói ngọng/lẫn lộn âm."""
        lower_word = word.lower()
        for prefix, replacement in sorted(CONSONANT_PAIRS.items(), key=lambda x: -len(x[0])):
            if lower_word.startswith(prefix) and self.rng.random() < self.consonant_swap_prob:
                is_upper = word[0].isupper()
                new_start = replacement.capitalize() if is_upper else replacement
                return new_start + word[len(prefix):]
        return word

    def shift_tone(self, word: str) -> str:
        """Đổi dấu sang dấu KHÁC — dấu vẫn còn, chỉ sai.

        Khác hẳn `remove_accents_from_word` (bỏ dấu hoàn toàn). Đây mới là lỗi phổ biến
        nhất của Whisper tiếng Việt: "làm giàu" -> "làm dầu", "củng cố" -> "cùng cố",
        "rực rỡ" -> "dực rỡ", "nồng nhiệt" -> "nông nhiệt", "dạy" -> "dậy".

        Hai cơ chế độc lập:
          - tone_shift  : đổi DẤU THANH (à<->a<->á<->ả<->ã<->ạ)
          - vowel_shift : đổi ÂM nguyên âm (a<->ă<->â, o<->ô<->ơ, u<->ư, e<->ê)

        CHỈ đổi TỐI ĐA MỘT ký tự mỗi từ. Nếu đổi nhiều ký tự cùng lúc sẽ sinh ra từ có
        hai dấu ("báo" -> "báỏ"), không tồn tại trong tiếng Việt — thứ mà ASR thật không
        tạo ra, nên đưa vào sẽ dạy mô hình loại lỗi hư cấu.
        """
        chars = list(word)
        # Chỉ đổi dấu ở ký tự ĐANG mang dấu thanh — không dán thêm dấu mới.
        tone_idx = [i for i, ch in enumerate(chars) if ch.lower() in _TONED_CHARS]
        vowel_idx = [i for i, ch in enumerate(chars) if ch.lower() in VOWEL_SHIFT_MAP]

        if tone_idx and self.rng.random() < self.tone_shift_prob:
            i = self.rng.choice(tone_idx)
            rep = self.rng.choice(TONE_SHIFT_MAP[chars[i].lower()])
            chars[i] = rep.upper() if chars[i].isupper() else rep
        elif vowel_idx and self.rng.random() < self.vowel_shift_prob:
            i = self.rng.choice(vowel_idx)
            rep = self.rng.choice(VOWEL_SHIFT_MAP[chars[i].lower()])
            chars[i] = rep.upper() if chars[i].isupper() else rep
        return "".join(chars)

    def apply_noise(self, clean_text: str) -> Tuple[str, List[str]]:
        """Nhận vào câu sạch và sinh ra câu nhiễu kèm nhãn các loại lỗi."""
        errors_applied = []

        # 0a. Câu SẠCH hoàn toàn (clean -> clean).
        #     Thiếu hẳn nhóm này là nguyên nhân trực tiếp của Edit-Precision 28%:
        #     mọi cặp huấn luyện đều có đầu vào sai, nên mô hình bị phạt mỗi lần
        #     không đổi gì và học thói quen "lúc nào cũng phải sửa" -> over-correction.
        if self.identity_prob > 0 and self.rng.random() < self.identity_prob:
            return clean_text, ["identity"]

        words = clean_text.split()
        noisy_words = []

        # 0b. MẤT TRẮN dấu thanh cả câu — đúng dạng đầu ra của Whisper khi bật chế độ
        #     không dấu / văn nói nhanh. Tập vàng của đề tài có 25/25 câu loại này,
        #     nhưng bộ nhiễu cũ chỉ tạo 0%. Đây là khoảng lệch train/test lớn nhất.
        if self.full_unaccent_prob > 0 and self.rng.random() < self.full_unaccent_prob:
            noisy_words = [self.remove_accents_from_word(w) for w in words]
            errors_applied.append("full_unaccent")
            result_text = " ".join(noisy_words)
            result_text = self.remove_punctuation(result_text)
            errors_applied.append("drop_punctuation")
            result_text = result_text.lower()
            errors_applied.append("lowercasing")
            return re.sub(r"\s+", " ", result_text).strip(), sorted(set(errors_applied))

        # 1. Nhiễu ở mức từ
        for w in words:
            curr_w = w
            # Xác suất bỏ dấu nguyên từ
            if self.rng.random() < self.drop_accent_word_prob:
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
                # Đổi dấu sang dấu KHÁC (lỗi Whisper: sai dấu, không mất dấu)
                new_w = self.shift_tone(curr_w)
                if new_w != curr_w:
                    errors_applied.append("tone_shift")
                    curr_w = new_w

            noisy_words.append(curr_w)

        result_text = " ".join(noisy_words)

        # 2. Xóa dấu câu
        if self.rng.random() < self.drop_punct_prob:
            result_text = self.remove_punctuation(result_text)
            errors_applied.append("drop_punctuation")

        # 3. Viết thường toàn bộ
        if self.rng.random() < self.lower_prob:
            result_text = result_text.lower()
            errors_applied.append("lowercasing")

        # Chuẩn hóa khoảng trắng
        result_text = re.sub(r"\s+", " ", result_text).strip()

        # Sắp xếp để nhãn lỗi ổn định giữa các tiến trình.
        # `list(set(...))` cho thứ tự phụ thuộc PYTHONHASHSEED, khiến tệp JSONL sinh ra
        # khác nhau về byte dù nội dung câu y hệt -> phá vỡ tái lập ở cấp tệp.
        return result_text, sorted(set(errors_applied))


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
