"""Dựng tập test cơ bản cho BÀI TOÁN GỐC mà đề cương đã tuyên bố.

Bối cảnh: đề cương Mục 2 hứa tập test 300-500 mẫu nhưng thực tế chỉ có 25 câu
(`data/gold_test/sample_test.jsonl`). 25 câu là quá ít: mỗi câu chiếm 4% chỉ số nên
không phân biệt được cải thiện thật với nhiễu đo. Script này mở rộng lên ~200 câu.

HAI ĐIỀU KIỆN BẮT BUỘC, đều đã trả giá trong dự án:

1. **Câu test phải CHƯA TỪNG dùng để train.** ShynBui trùng lặp nội bộ ~21%, nên lấy
   theo index là không đủ — phải anti-join theo `text` với đúng tập đã nạp cho train.

2. **Nhiễu phải KHÁC cấu hình train.** Nếu sinh test bằng chính bộ nhiễu + cấu hình
   đã dùng để train thì đang đo "mô hình thuộc lòng thói quen hàm nhiễu nhà mình",
   không phải năng lực sửa lỗi. Script này dùng `profile` riêng cho test.

Kết quả:
    data/gold_test/gd1_basic/basic_test_unaccent_all.jsonl   200 câu, mất trắng dấu
    data/gold_test/gd1_basic/basic_test_asr_hard.jsonl       200 câu, nhiễu hỗn hợp

Chạy:
    python -m src.data_processing.build_basic_test
"""

import argparse
import json
import sys
from pathlib import Path

from .corpus_builder import dedupe_sentences, make_pairs

# Số câu đã dùng cho train Lần 5 (nạp dư rồi cắt còn 100k sạch). Mọi câu nằm trong
# vùng này KHÔNG được xuất hiện trong tập test.
TRAIN_FETCH = 220_000


def main() -> int:
    parser = argparse.ArgumentParser(description="Dựng tập test cơ bản (bài toán gốc)")
    parser.add_argument("--n", type=int, default=200, help="Số câu mỗi tập test")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", default="data/gold_test/gd1_basic")
    parser.add_argument("--profiles", nargs="*", default=["unaccent_all", "asr_hard"])
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Nạp ShynBui train[:{TRAIN_FETCH + 60_000:,}] để tách vùng train / vùng test ...")
    raw = load_dataset("ShynBui/Vietnamese_spelling_error", split=f"train[:{TRAIN_FETCH + 60_000}]")
    all_texts = [(r["text"] or "").strip() for r in raw]

    train_pool, test_pool = all_texts[:TRAIN_FETCH], all_texts[TRAIN_FETCH:]
    print(f"  vùng train: {len(train_pool):,} câu | vùng ứng viên test: {len(test_pool):,} câu")

    # Anti-join theo NỘI DUNG, không theo index
    train_set = {t.lower() for t in train_pool}
    test_clean, stats = dedupe_sentences(test_pool)
    before = len(test_clean)
    test_clean = [t for t in test_clean if t.lower() not in train_set]
    print(f"  khử trùng lặp: {before:,} -> loại tiếp {before - len(test_clean):,} câu trùng vùng train")
    print(f"  còn đủ điều kiện: {len(test_clean):,} câu")

    if len(test_clean) < args.n:
        print(f"[LỖI] Chỉ có {len(test_clean):,} câu, cần {args.n}. Giảm --n hoặc nạp thêm.")
        return 1

    # Cắt TRẢI ĐỀU (bước nhảy) thay vì lấy 200 câu đầu, để không dồn vào một vùng dữ liệu
    step = len(test_clean) // args.n
    picked = [test_clean[i * step] for i in range(args.n)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for profile in args.profiles:
        rows = make_pairs(picked, profile=profile, tag=f"basic_{profile}", seed=args.seed)
        # Đổi tên trường cho khớp định dạng các tập test khác
        for i, r in enumerate(rows, 1):
            r["id"] = i
            r["source"] = f"synthetic_{profile}"

        path = out_dir / f"basic_test_{profile}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        changed = sum(1 for r in rows if r["noisy"] != r["clean"])
        print(f"\nGhi {path}: {len(rows)} câu ({changed} câu bị đổi bởi '{profile}')")

    print("\nXong. Tập test này dùng cho run_benchmark.py (KHÔNG phải cho train).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
