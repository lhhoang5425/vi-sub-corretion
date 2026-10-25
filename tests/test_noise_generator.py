"""Kiểm thử mô-đun sinh dữ liệu nhiễu."""

from src.data_processing.noise_generator import ASRNoiseGenerator


def test_noise_generator_applies_noise():
    gen = ASRNoiseGenerator(
        drop_punct_prob=1.0,
        lower_prob=1.0,
        consonant_swap_prob=0.0,
        tone_swap_prob=0.0,
        drop_accent_word_prob=0.0,
    )
    clean = "Hôm nay thời tiết rất đẹp!"
    noisy, errors = gen.apply_noise(clean)

    assert "!" not in noisy
    assert noisy == noisy.lower()
    assert "drop_punctuation" in errors
    assert "lowercasing" in errors
