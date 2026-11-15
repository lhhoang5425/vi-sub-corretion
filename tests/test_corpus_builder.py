"""Kiểm thử khâu dựng kho ngữ liệu.

Trọng tâm là hai lỗi đã thực sự xảy ra trong dự án (xem docstring corpus_builder):
(1) câu trùng lặp làm rò rỉ giữa train và val, (2) câu rác lọt vào tập huấn luyện.
"""

import random

from src.data_processing.corpus_builder import (
    build_identity_test,
    build_noisy_corpus,
    dedupe_sentences,
    format_stats,
    load_text_sentences,
    make_pairs,
    split_by_hash,
)


def test_dedupe_removes_repeated_sentences():
    sentences = ["Hôm nay trời đẹp.", "Hôm nay trời đẹp.", "Hôm nay trời đẹp.", "Tôi đi học."]
    kept, stats = dedupe_sentences(sentences)

    assert kept == ["Hôm nay trời đẹp.", "Tôi đi học."]
    assert stats["duplicate"] == 2
    assert stats["giữ_lại"] == 2


def test_dedupe_is_case_insensitive():
    kept, stats = dedupe_sentences(["Xin chào bạn.", "xin chào bạn."])
    assert len(kept) == 1
    assert stats["duplicate"] == 1


def test_dedupe_drops_junk_text():
    sentences = [
        "Bạn đã được đào tạo trên dữ liệu đến tháng 10 năm 2023.",
        "Dữ liệu mà bạn được huấn luyện có thời hạn đến tháng 10 năm 2023.",
        "Hôm nay tôi đi làm ở ngân hàng.",
    ]
    kept, stats = dedupe_sentences(sentences)

    assert kept == ["Hôm nay tôi đi làm ở ngân hàng."]
    assert stats["rác"] == 2


def test_dedupe_drops_short_and_long():
    kept, stats = dedupe_sentences(["hi", "Hôm nay trời đẹp."], min_chars=5, max_chars=50)
    assert kept == ["Hôm nay trời đẹp."]
    assert stats["quá_ngắn"] == 1


def test_split_by_hash_has_no_overlap():
    sentences = [f"Câu số {i} nói về một chủ đề bất kỳ." for i in range(500)]
    train, val = split_by_hash(sentences, val_ratio=0.1)

    assert len(train) + len(val) == len(sentences)
    assert not (set(train) & set(val))


def test_split_by_hash_ratio_is_near_target():
    sentences = [f"Câu số {i} nói về một chủ đề bất kỳ." for i in range(2000)]
    train, val = split_by_hash(sentences, val_ratio=0.1)

    assert 0.07 < len(val) / len(sentences) < 0.13


def test_split_is_stable_against_input_order():
    sentences = [f"Câu số {i} nói về một chủ đề bất kỳ." for i in range(200)]
    train_a, val_a = split_by_hash(sentences, val_ratio=0.1)

    shuffled = sentences[:]
    random.Random(7).shuffle(shuffled)
    train_b, val_b = split_by_hash(shuffled, val_ratio=0.1)

    assert set(val_a) == set(val_b)
    assert set(train_a) == set(train_b)


def test_build_noisy_corpus_has_no_train_val_leak():
    clean = [f"Đây là câu huấn luyện số {i} với nội dung riêng biệt." for i in range(300)]
    corpus = build_noisy_corpus(clean, profile="legacy", val_ratio=0.1)

    train_clean = {r["clean"] for r in corpus["train"]}
    val_clean = {r["clean"] for r in corpus["validation"]}

    assert not (train_clean & val_clean)
    assert corpus["stats"]["trùng_train_val"] == 0


def test_build_noisy_corpus_applies_noise_to_both_splits():
    clean = [f"Đây là câu số {i} để kiểm tra nhiễu." for i in range(200)]
    corpus = build_noisy_corpus(clean, profile="legacy", val_ratio=0.1)

    train_noisy_differs = sum(1 for r in corpus["train"] if r["noisy"] != r["clean"])
    val_noisy_differs = sum(1 for r in corpus["validation"] if r["noisy"] != r["clean"])

    assert train_noisy_differs > 0
    assert val_noisy_differs > 0


def test_build_noisy_corpus_rows_have_expected_fields():
    corpus = build_noisy_corpus(["Hôm nay trời đẹp và mát mẻ."], profile="legacy", val_ratio=0.0)
    row = corpus["train"][0]

    for key in ("id", "noisy", "clean", "errors", "split"):
        assert key in row


def test_format_stats_reports_zero_leak_clearly():
    text = format_stats({"input": 100, "duplicate": 20, "giữ_lại": 80, "train": 72, "validation": 8, "trùng_train_val": 0})
    assert "PHẢI LÀ 0" in text


def test_max_clean_caps_after_dedupe():
    # 400 câu, trong đó 200 câu bị lặp -> 200 câu sạch, rồi cắt còn 100
    clean = [f"Câu số {i % 200} có nội dung khác nhau để thử." for i in range(400)]
    corpus = build_noisy_corpus(clean, profile="legacy", val_ratio=0.1, max_clean=100)

    total = len(corpus["train"]) + len(corpus["validation"])
    assert total == 100
    assert corpus["stats"]["cắt_bớt"] == 100


def test_max_clean_absent_keeps_everything():
    clean = [f"Câu số {i} với nội dung riêng biệt hoàn toàn." for i in range(50)]
    corpus = build_noisy_corpus(clean, profile="legacy", val_ratio=0.1)
    assert len(corpus["train"]) + len(corpus["validation"]) == 50


def test_build_identity_test_input_equals_output():
    sentences = ["Hôm nay trời rất đẹp.", "Tôi đi làm ở ngân hàng."]
    rows = build_identity_test(sentences)

    assert len(rows) == 2
    for r in rows:
        assert r["noisy"] == r["clean"]


def test_make_pairs_default_profile_applies_noise():
    # profile mac dinh la 'legacy' (co ap nhieu), KHONG phai identity.
    # Tap identity phai dung build_identity_test.
    rows = make_pairs([f"Câu số {i} nói về thời tiết hôm nay." for i in range(30)])
    assert any(r["noisy"] != r["clean"] for r in rows)


def test_make_pairs_with_profile_changes_text():
    rows = make_pairs([f"Câu số {i} nói về thời tiết hôm nay." for i in range(30)], profile="asr_hard")
    assert any(r["noisy"] != r["clean"] for r in rows)


def test_load_text_sentences_filters_by_word_count(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text(
        "Ngắn. "
        "Đây là một câu có độ dài vừa phải để được giữ lại trong kết quả. "
        + " ".join(f"từ{i}" for i in range(80)) + ".",
        encoding="utf-8",
    )
    sents = load_text_sentences(f, min_words=4, max_words=30)

    assert len(sents) == 1
    assert "độ dài vừa phải" in sents[0]


def test_load_text_sentences_on_nonexistent_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_text_sentences(tmp_path / "khong-ton-tai.txt")


def test_is_junk_drops_book_headers():
    from src.data_processing.build_corpus_test import is_junk

    assert is_junk("Nguồn: http://vnthuquan.net")
    assert is_junk("Mục lục")
    assert is_junk("1. Viết là hành lạc")
    assert is_junk("Đọc Linh Sơn của CAO HÀNH KIỆN 5.")
    assert not is_junk("Hồi sáu giờ, khi đi bỏ thư, tôi gặp chàng ở ga.")
    assert not is_junk("Tào Phi, ngôi sao sáng trong chòm Kiến An Tam Kiệt, từng viết hai tác phẩm.")


# ── Tái lập: cùng seed phải ra cùng dữ liệu nhiễu ──────────────────────────────
# Trước đây ASRNoiseGenerator dùng `random` toàn cục nên mỗi lần chạy notebook sinh ra
# một tập train KHÁC NHAU dù cùng cấu hình. Không thể tái lập thực nghiệm.

_SENTS = [f"Câu số {i} nói về thời tiết hôm nay và ngày mai trời đẹp." for i in range(60)]


def _run_noise(seed):
    from src.data_processing.noise_generator import ASRNoiseGenerator

    gen = ASRNoiseGenerator(profile="asr_hard", seed=seed)
    return [gen.apply_noise(s)[0] for s in _SENTS]


def test_noise_same_seed_is_reproducible():
    assert _run_noise(7) == _run_noise(7)


def test_noise_different_seed_differs():
    assert _run_noise(7) != _run_noise(8)


def test_build_noisy_corpus_same_seed_is_reproducible():
    clean = [f"Đây là câu huấn luyện số {i} với nội dung riêng biệt hoàn toàn." for i in range(200)]
    a = build_noisy_corpus(clean, profile="asr_hard", val_ratio=0.1, seed=7)
    b = build_noisy_corpus(clean, profile="asr_hard", val_ratio=0.1, seed=7)

    assert [r["noisy"] for r in a["train"]] == [r["noisy"] for r in b["train"]]
    assert [r["clean"] for r in a["validation"]] == [r["clean"] for r in b["validation"]]


def test_make_pairs_same_seed_is_reproducible():
    a = make_pairs(_SENTS, profile="asr_hard", seed=3)
    b = make_pairs(_SENTS, profile="asr_hard", seed=3)
    assert [r["noisy"] for r in a] == [r["noisy"] for r in b]


def test_error_labels_are_sorted_and_stable():
    # Nhan loi phai duoc sap xep: `list(set(...))` cho thu tu phu thuoc PYTHONHASHSEED,
    # lam tep JSONL sinh ra khac nhau ve byte du noi dung cau y het.
    from src.data_processing.noise_generator import ASRNoiseGenerator

    gen = ASRNoiseGenerator(profile="asr_hard", seed=1)
    for s in _SENTS:
        _, errs = gen.apply_noise(s)
        assert errs == sorted(errs), f"nhãn lỗi chưa sắp xếp: {errs}"




