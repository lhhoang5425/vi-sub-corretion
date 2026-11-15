"""Kiểm thử các hồ sơ nhiễu trong noise_generator.

Trọng tâm là `whisper_like` — hồ sơ mô phỏng lỗi THẬT của Whisper tiếng Việt, đo từ
`data/raw/kichban_2026-10-25.srt`. Khác biệt cốt lõi so với các hồ sơ trước: Whisper
GIỮ dấu thanh (99,3% câu) và sai dấu, chứ không bỏ dấu.
"""

import re

from src.data_processing.noise_generator import ASRNoiseGenerator, NOISE_PROFILES

TONE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)

SENTS = [f"Câu số {i} nói về hoạt động của Ban Thanh tra Nhân dân và báo cáo tóm lược."
         for i in range(120)]


def test_all_profiles_exist():
    for name in ("legacy", "asr_hard", "unaccent_all", "whisper_like"):
        assert name in NOISE_PROFILES


def test_whisper_like_keeps_tone_marks():
    """Điểm khác biệt cốt lõi: KHÔNG câu nào mất sạch dấu thanh."""
    gen = ASRNoiseGenerator(profile="whisper_like", seed=5)
    outs = [gen.apply_noise(s)[0] for s in SENTS]
    n_with_tone = sum(1 for o in outs if TONE.search(o))
    assert n_with_tone == len(outs), f"chỉ {n_with_tone}/{len(outs)} câu còn dấu thanh"


def test_unaccent_all_strips_every_tone():
    gen = ASRNoiseGenerator(profile="unaccent_all", seed=5)
    outs = [gen.apply_noise(s)[0] for s in SENTS]
    assert not any(TONE.search(o) for o in outs)


def test_whisper_like_differs_from_unaccent_all():
    a = ASRNoiseGenerator(profile="whisper_like", seed=5)
    b = ASRNoiseGenerator(profile="unaccent_all", seed=5)
    assert [a.apply_noise(s)[0] for s in SENTS] != [b.apply_noise(s)[0] for s in SENTS]


def test_whisper_like_keeps_a_share_of_clean_sentences():
    """File ASR thật có 54,5% block đúng hoàn toàn. Hồ sơ phải tái tạo nhóm đó."""
    gen = ASRNoiseGenerator(profile="whisper_like", seed=3)
    outs = [gen.apply_noise(s)[0] for s in SENTS]
    identity = sum(1 for o, s in zip(outs, SENTS) if o == s) / len(SENTS)
    assert 0.25 < identity < 0.70, f"tỷ lệ giữ nguyên = {identity:.2f}, ngoài khoảng mong đợi"


def test_shift_tone_changes_at_most_one_char_per_word():
    """Đổi nhiều ký tự cùng lúc sẽ sinh từ có hai dấu ('báo' -> 'báỏ') — không có thật."""
    gen = ASRNoiseGenerator(profile="whisper_like", seed=1)
    for w in ["báo", "hoạt", "nghiêng", "nguyễn", "thuyền", "Nhân", "dân"]:
        out = gen.shift_tone(w)
        assert len(out) == len(w)
        diffs = sum(1 for a, b in zip(w, out) if a != b)
        assert diffs <= 1, f"{w} -> {out} ({diffs} ký tự bị đổi)"


def test_shift_tone_produces_no_double_tone_mark():
    """Không được có HAI dấu thanh trong cùng một từ.

    Lưu ý phân biệt: dấu móc/mũ (ơ, ư, â, ê, ô, ă) KHÔNG phải dấu thanh. Từ hợp lệ
    như "được" có 2 ký tự mang dấu (ư, ợ) nhưng chỉ một dấu THANH. Nên chỉ đếm dấu thanh.
    """
    tone_mark = re.compile(r"[àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")
    gen = ASRNoiseGenerator(profile="whisper_like", seed=9)
    for _ in range(300):
        out = gen.shift_tone("hoạt động báo cáo")
        for word in out.split():
            marks = tone_mark.findall(word)
            assert len(marks) <= 1, f"'{word}' có {len(marks)} dấu thanh"


def test_shift_tone_is_disabled_when_prob_zero():
    gen = ASRNoiseGenerator(profile="legacy", seed=1)
    assert gen.shift_tone("báo cáo") == "báo cáo"
