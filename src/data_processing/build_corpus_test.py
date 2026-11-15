"""Dựng tập test từ kho văn bản người viết trong `corpus/`.

Vì sao cần tập test riêng ngoài `sample_test.jsonl`:

1. **Tập test hiện tại không có câu sạch nào.** Nên phép đo over-correction — câu vốn đã
   đúng mà bị mô hình sửa nhầm — luôn rỗng. Kho `corpus/` toàn văn bản đã đúng chính tả,
   nên dùng thẳng làm tập **identity**: đầu vào bằng đầu ra, mọi thay đổi đều sai.

2. **Ngoài miền huấn luyện.** Câu trong `corpus/` là văn học dịch (trung bình ~35 từ),
   khác hẳn câu ngắn dạng hội thoại của ShynBui (~27 token). Nếu mô hình chỉ tốt trên
   loại câu nó từng thấy mà phá câu văn dài, phép thử này phát hiện ra.

Tạo hai tệp:
    data/gold_test/control/corpus_identity.jsonl  — đầu vào = đầu ra = câu đúng  -> đo over-correction
    data/gold_test/control/corpus_noisy.jsonl     — sinh nhiễu từ cùng câu         -> đo khôi phục (ngoài miền)

Chạy:
    python -m src.data_processing.build_corpus_test
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from .corpus_builder import (
    build_identity_test,
    load_text_sentences,
    make_pairs,
    write_jsonl,
)


def hash_sentences(sentences) -> str:
    """Băm nội dung tập câu, để chứng minh tập test tái lập được mà không cần commit văn bản."""
    h = hashlib.sha256()
    for s in sentences:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# Dòng không phải câu văn: tiêu đề đầu sách, mục lục, ghi nguồn của các tệp .txt.
JUNK_HEADER = re.compile(
    r"vnthuquan|mục lục|nguồn:|phát hành:|^\d+\.\s",
    re.IGNORECASE,
)


def is_junk(sentence: str) -> bool:
    """Loại dòng tiêu đề / mục lục.

    Ngoài các mẫu cố định, loại thêm câu NGẮN mà kết thúc bằng "<số>." — dạng mục lục
    kiểu "Đọc Linh Sơn của CAO HÀNH KIỆN 5.". Điều kiện ngắn (< 12 từ) để không loại
    nhầm câu văn thật kết thúc bằng năm tháng.
    """
    if JUNK_HEADER.search(sentence):
        return True
    return len(sentence.split()) < 12 and bool(re.search(r"\s\d+\.$", sentence))


def main() -> int:
    parser = argparse.ArgumentParser(description="Dựng tập test từ corpus/")
    parser.add_argument("--corpus_dir", default="corpus", help="Thư mục chứa các tệp .txt")
    parser.add_argument("--out_dir", default="data/gold_test/control", help="Nơi ghi tệp .jsonl")
    parser.add_argument("--profile", default="asr_hard", help="Hồ sơ nhiễu cho tệp thứ hai")
    parser.add_argument("--min_words", type=int, default=8)
    parser.add_argument("--max_words", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42, help="Seed sinh nhiễu, ghi vào manifest để tái lập")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)

    if not corpus_dir.is_dir():
        print(f"[LỖI] Không thấy thư mục {corpus_dir}")
        return 1

    sentences = []
    files = sorted(p for p in corpus_dir.glob("*.txt"))
    if not files:
        print(f"[LỖI] Không có tệp .txt nào trong {corpus_dir}")
        return 1

    print(f"Đọc {len(files)} tệp trong {corpus_dir}/")
    for f in files:
        part = load_text_sentences(f, min_words=args.min_words, max_words=args.max_words)
        sentences.extend(part)
        print(f"  {f.name[:52]:<54}{len(part):>6,} câu")

    # Khử trùng lặp giữa các tệp, loại dòng tiêu đề/mục lục
    uniq, seen, dropped_header = [], set(), 0
    for s in sentences:
        if is_junk(s):
            dropped_header += 1
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)

    print(f"\nTổng: {len(sentences):,} câu -> {len(uniq):,} câu sau khử trùng lặp"
          f" (loại {dropped_header} dòng tiêu đề/mục lục)")

    id_rows = build_identity_test(uniq)
    noisy_rows = make_pairs(uniq, profile=args.profile, tag="corpus_ood", seed=args.seed)

    n1 = write_jsonl(id_rows, out_dir / "corpus_identity.jsonl")
    n2 = write_jsonl(noisy_rows, out_dir / "corpus_noisy.jsonl")

    changed = sum(1 for r in noisy_rows if r["noisy"] != r["clean"])
    print(f"\nGhi {out_dir}/corpus_identity.jsonl : {n1:,} câu (đầu vào = đầu ra, đo over-correction)")
    print(f"Ghi {out_dir}/corpus_noisy.jsonl    : {n2:,} câu ({changed:,} câu bị đổi bởi hồ sơ '{args.profile}')")

    # Manifest: ĐƯỢC commit (khác với hai tệp .jsonl ở trên, bị gitignore vì chứa
    # nguyên văn bản có bản quyền). Manifest chỉ chứa hash + thống kê, đủ để chứng
    # minh tập test sinh lại là y hệt.
    manifest = {
        "tao_luc": datetime.now().astimezone().isoformat(timespec="seconds"),
        "so_cau": len(uniq),
        "so_tep_nguon": len(files),
        "tep_nguon": [f.name for f in files],
        "hash_clean_sha256": hash_sentences(uniq),
        "ho_so_nhieu": args.profile,
        "seed_nhieu": args.seed,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "ghi_chu": (
            "Tập test sinh từ văn bản có bản quyền nên KHÔNG commit corpus/ và corpus_*.jsonl. "
            "Manifest này để chứng minh tái lập: chạy lại build_corpus_test trên cùng corpus/ "
            "phải cho ra đúng hash_clean_sha256 và số câu này."
        ),
    }
    mpath = out_dir / "corpus_test_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ghi {mpath}  (hash: {manifest['hash_clean_sha256'][:16]}...)")

    print("\nLƯU Ý: nên anti-join theo trường `clean` với tập huấn luyện trước khi dùng chính thức.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
