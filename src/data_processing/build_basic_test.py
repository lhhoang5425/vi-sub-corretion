"""Dựng tập test cho BÀI TOÁN GỐC (gd1): văn bản mất dấu.

Hai điều kiện bắt buộc, đều đã trả giá trong dự án:

1. Câu test phải CHƯA TỪNG dùng để train. ShynBui trùng lặp nội bộ ~21%, nên lấy theo
   index là không đủ — phải anti-join theo `text` với đúng tập đã nạp cho train.
2. Nhiễu phải KHÁC cấu hình train. Nếu sinh test bằng chính bộ nhiễu + cấu hình đã
   dùng để train thì đang đo "mô hình thuộc lòng hàm nhiễu nhà mình", không phải năng lực.

Kết quả:
    basic_test_unaccent_all.jsonl   200 câu, 100% mất dấu     (một nguồn: ShynBui)
    basic_test_asr_hard.jsonl       200 câu, nhiễu hỗn hợp    (một nguồn: ShynBui)
    basic_test_multisource.jsonl    300 câu, HAI nguồn        (ShynBui + Wikipedia)

Tập cuối để phát hiện mô hình chỉ giỏi văn phong một nguồn: run_benchmark tách riêng
điểm theo trường `source`.

Chạy:
    python -m src.data_processing.build_basic_test
"""

import argparse
import json
import sys
from pathlib import Path

from .corpus_builder import dedupe_sentences, load_wikipedia_sentences, make_pairs

# Số câu đã dùng cho train Lần 5 (nạp dư rồi cắt còn 100k sạch). Mọi câu nằm trong
# vùng này KHÔNG được xuất hiện trong tập test.
TRAIN_FETCH = 220_000


def main() -> int:
    parser = argparse.ArgumentParser(description="Dựng tập test cơ bản (bài toán gốc)")
    parser.add_argument("--n", type=int, default=200, help="Số câu mỗi tập một nguồn")
    parser.add_argument("--wiki_n", type=int, default=150, help="Số câu Wikipedia cho tập đa nguồn")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", default="data/gold_test/gd1_basic")
    parser.add_argument("--profiles", nargs="*", default=["unaccent_all", "asr_hard"])
    parser.add_argument("--skip_wiki", action="store_true", help="Bỏ qua phần Wikipedia")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Nạp ShynBui train[:{TRAIN_FETCH + 60_000:,}] để tách vùng train / vùng test ...")
    raw = load_dataset("ShynBui/Vietnamese_spelling_error", split=f"train[:{TRAIN_FETCH + 60_000}]")
    all_texts = [(r["text"] or "").strip() for r in raw]

    train_pool, test_pool = all_texts[:TRAIN_FETCH], all_texts[TRAIN_FETCH:]
    print(f"  vùng train: {len(train_pool):,} câu | vùng ứng viên test: {len(test_pool):,} câu")

    train_set = {t.lower() for t in train_pool}
    test_clean, _ = dedupe_sentences(test_pool)
    before = len(test_clean)
    test_clean = [t for t in test_clean if t.lower() not in train_set]
    print(f"  khử trùng lặp: {before:,} -> loại tiếp {before - len(test_clean):,} câu trùng vùng train")

    if len(test_clean) < args.n:
        print(f"[LỖI] Chỉ có {len(test_clean):,} câu, cần {args.n}. Giảm --n hoặc nạp thêm.")
        return 1

    # Cắt TRẢI ĐỀU thay vì lấy đầu, để không dồn vào một vùng dữ liệu
    step = len(test_clean) // args.n
    picked = [test_clean[i * step] for i in range(args.n)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows) -> None:
        for i, r in enumerate(rows, 1):
            r["id"] = i
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        changed = sum(1 for r in rows if r["noisy"] != r["clean"])
        print(f"  Ghi {name}: {len(rows)} câu ({changed} câu bị đổi)")

    for profile in args.profiles:
        rows = make_pairs(picked, profile=profile, tag=f"basic_{profile}",
                          seed=args.seed, source="shynbui")
        write(f"basic_test_{profile}.jsonl", rows)

    if not args.skip_wiki and args.wiki_n > 0:
        print(f"\nLấy {args.wiki_n} câu từ Wikipedia tiếng Việt (CC BY-SA, commit được) ...")
        wiki = load_wikipedia_sentences(args.wiki_n, seed=args.seed)
        wiki = [s for s in wiki if s.lower() not in train_set]
        print(f"  lấy được {len(wiki)} câu")

        rows = make_pairs(picked[:args.wiki_n], profile="unaccent_all",
                          tag="multisource", seed=args.seed, source="shynbui")
        rows += make_pairs(wiki, profile="unaccent_all",
                           tag="multisource", seed=args.seed + 1, source="wikipedia")
        write("basic_test_multisource.jsonl", rows)

    print("\nXong. Tập test dùng cho run_benchmark.py (KHÔNG phải cho train).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
