"""Xây kho ngữ liệu huấn luyện từ câu sạch: khử trùng lặp, lọc rác, chia tập, sinh nhiễu.

Tách riêng khỏi `dataset_loader` (chỉ lo đọc tệp và tokenize) vì đây là logic DỮ LIỆU,
và vì hai lý do đã trả giá trong dự án:

1. **Dataset nguồn trùng lặp ~21%.** `ShynBui/Vietnamese_spelling_error` có 100.000 câu
   nhưng chỉ 78.985 nội dung khác nhau. Nếu chia tập theo *vị trí dòng* (`i % 10`), cùng
   một câu có bản rơi vào train và bản rơi vào val — đo được 20.8% val trùng train. Khi đó
   `eval_loss` thấp một phần vì mô hình thuộc bài, không phải vì khái quát hóa tốt.

2. **Dataset lẫn văn bản không phải tiếng Việt tự nhiên.** Một số câu mô tả mô hình AI
   xuất hiện hàng chục lần (nhiều nhất 60 lần), làm lệch phân bố nội dung.

Thứ tự bắt buộc: **khử trùng lặp → lọc rác → chia tập → mới sinh nhiễu.** Không được đảo.
"""

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .noise_generator import ASRNoiseGenerator

# Câu rác: văn bản mô tả mô hình AI, không phải tiếng Việt tự nhiên.
# Đo được các câu này lặp 60 / 43 lần trong 100.000 câu đầu của dataset nguồn.
DEFAULT_JUNK_PATTERNS = [
    r"được huấn luyện có thời hạn",
    r"được đào tạo trên dữ liệu đến",
    r"thời hạn đến tháng \d+ năm \d+",
    r"dữ liệu mà bạn được huấn luyện",
    r"tôi là một mô hình ngôn ngữ",
]


def _bucket_of(text: str, n_buckets: int = 1000) -> int:
    """Băm nội dung câu thành một số ổn định trong [0, n_buckets).

    Dùng hash nội dung thay vì vị trí dòng để việc chia tập không phụ thuộc vào thứ tự
    câu trong tệp và không thay đổi giữa các lần chạy.
    """
    digest = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def dedupe_sentences(
    sentences: Sequence[str],
    min_chars: int = 5,
    max_chars: int = 512,
    drop_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Khử trùng lặp và lọc câu rác, trả về (danh sách câu, thống kê).

    Giữ lại BẢN GHI ĐẦU TIÊN của mỗi nội dung. Thống kê trả về để in ra log — số liệu
    bị loại cần thấy được, không nên im lặng.
    """
    patterns = [re.compile(p, re.IGNORECASE) for p in (drop_patterns if drop_patterns is not None else DEFAULT_JUNK_PATTERNS)]

    seen = set()
    kept: List[str] = []
    stats = {"input": len(sentences), "duplicate": 0, "quá_ngắn": 0, "quá_dài": 0, "rác": 0, "giữ_lại": 0}

    for raw in sentences:
        text = (raw or "").strip()

        if len(text) < min_chars:
            stats["quá_ngắn"] += 1
            continue
        if len(text) > max_chars:
            stats["quá_dài"] += 1
            continue
        if any(p.search(text) for p in patterns):
            stats["rác"] += 1
            continue

        key = text.lower()
        if key in seen:
            stats["duplicate"] += 1
            continue

        seen.add(key)
        kept.append(text)

    stats["giữ_lại"] = len(kept)
    return kept, stats


def split_by_hash(
    sentences: Sequence[str],
    val_ratio: float = 0.1,
    n_buckets: int = 1000,
) -> Tuple[List[str], List[str]]:
    """Chia train/val theo băm nội dung, không theo vị trí dòng.

    Sau khi `dedupe_sentences` chạy, hai tập chắc chắn không giao nhau về nội dung.
    Dùng hash còn bảo đảm cùng một câu luôn rơi vào cùng một tập dù thứ tự đầu vào đổi.
    """
    cut = int(round(val_ratio * n_buckets))
    train = [s for s in sentences if _bucket_of(s, n_buckets) >= cut]
    val = [s for s in sentences if _bucket_of(s, n_buckets) < cut]
    return train, val


def build_noisy_corpus(
    clean_sentences: Sequence[str],
    profile: str = "legacy",
    val_ratio: float = 0.1,
    min_chars: int = 5,
    drop_patterns: Optional[Sequence[str]] = None,
    seed: int = 42,
    max_clean: Optional[int] = None,
    noise_gen: Optional[ASRNoiseGenerator] = None,
) -> Dict[str, List[Dict]]:
    """Từ kho câu sạch, dựng kho cặp `(noisy, clean)` cho train và val.

    Trả về dict: `{"train": [...], "validation": [...], "stats": {...}}`.
    Mỗi phần tử là dict có `noisy`, `clean`, `errors` — đúng định dạng JSONL mà
    `dataset_loader.load_correction_dataset` đọc được.

    `max_clean`: giới hạn số câu sạch SAU khi khử trùng lặp. Dùng khi nguồn có trùng lặp
    nội bộ cao: nạp dư (ví dụ 200.000 câu) rồi cắt còn đúng số mong muốn (100.000 câu
    sạch) thay vì nhận về số ít hơn do bị loại bản sao. Cắt SAU khi trộn ngẫu nhiên
    (seed cố định) để không chỉ lấy một vùng dữ liệu.

    Lưu ý về val: val **cũng được áp nhiễu**, cùng hồ sơ nhiễu với train. Đó là điều
    kiện bắt buộc để val đo được khả năng khái quát hóa — val sạch mà train nhiễu thì
    con số đo ra vô nghĩa.
    """
    import random as _random

    rng = _random.Random(seed)
    gen = noise_gen or ASRNoiseGenerator(profile=profile, seed=seed)

    kept, stats = dedupe_sentences(clean_sentences, min_chars=min_chars, drop_patterns=drop_patterns)

    if max_clean is not None and len(kept) > max_clean:
        rng.shuffle(kept)
        stats["cắt_bớt"] = len(kept) - max_clean
        kept = kept[:max_clean]
    stats["giữ_lại"] = len(kept)

    train_clean, val_clean = split_by_hash(kept, val_ratio=val_ratio)

    def make(rows: Sequence[str], tag: str) -> List[Dict]:
        out = []
        for i, s in enumerate(rows):
            noisy, errs = gen.apply_noise(s)
            out.append({
                "id": i + 1,
                "noisy": noisy,
                "clean": s,
                "errors": errs,
                "split": tag,
            })
        return out

    train_rows = make(train_clean, "train")
    val_rows = make(val_clean, "validation")

    # Chốt chặn: train và val tuyệt đối không được trùng nội dung
    overlap = set(x["clean"] for x in train_rows) & set(x["clean"] for x in val_rows)
    if overlap:
        raise RuntimeError(
            f"RÒ RỈ DỮ LIỆU: {len(overlap)} câu vừa ở train vừa ở val. "
            "Kiểm tra lại split_by_hash trước khi train."
        )

    stats["train"] = len(train_rows)
    stats["validation"] = len(val_rows)
    stats["nhiễu_train"] = gen.__class__.__name__ + (f"({profile})" if profile else "")
    stats["trùng_train_val"] = len(overlap)

    return {"train": train_rows, "validation": val_rows, "stats": stats}


def load_text_sentences(path, min_words: int = 4, max_words: int = 60) -> List[str]:
    """Đọc một tệp .txt thuần (truyện, bài báo...) và tách thành danh sách câu.

    Dùng cho kho `corpus/` khi muốn lấy văn bản NGƯỜI VIẾT (đã đúng chính tả) làm
    dữ liệu test ngoài miền huấn luyện. Lọc theo số từ để tránh câu quá ngắn (vô nghĩa)
    và quá dài (vượt max_length của mô hình).
    """
    import re
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    raw = re.split(r"(?<=[.!?…])\s+", text)
    out = []
    for s in raw:
        s = " ".join(s.split())
        if len(s) < 15:
            continue
        n = len(s.split())
        if min_words <= n <= max_words:
            out.append(s)
    return out


def load_wikipedia_sentences(
    n_sentences: int,
    min_words: int = 8,
    max_words: int = 60,
    lang: str = "vi",
    seed: int = 42,
) -> List[str]:
    """Lấy câu sạch từ Wikipedia tiếng Việt (giấy phép CC BY-SA, commit lên git được).

    Dùng làm NGUỒN THỨ HAI cho tập test, để không phụ thuộc độc quyền vào ShynBui.
    Nếu mô hình chỉ giỏi văn phong của một nguồn, bảng theo `source` sẽ lộ ra.

    Đọc ở chế độ streaming nên không tải cả kho về đĩa — dừng ngay khi đủ câu.
    """
    import re

    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                      split="train", streaming=True)
    seen = set()
    out: List[str] = []
    for article in ds:
        for s in re.split(r"(?<=[.!?…])\s+", article.get("text") or ""):
            s = " ".join(s.split())
            if len(s) < 20 or s.count("(") or s.count("="):
                continue
            if not (min_words <= len(s.split()) <= max_words):
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= n_sentences:
                return out
    return out


def make_pairs(
    sentences: Sequence[str],
    profile: str = "legacy",
    tag: str = "test",
    seed: int = 42,
    noise_gen: Optional[ASRNoiseGenerator] = None,
    source: str = "unknown",
) -> List[Dict]:
    """Sinh cặp `(noisy, clean)` cho một danh sách câu, KHÔNG chia train/val.

    Dùng để dựng tập test. `profile` mặc định là `"legacy"` để không có chỗ mơ hồ —
    muốn tập identity (đầu vào = đầu ra) thì dùng `build_identity_test`, không dùng
    hàm này với một giá trị đặc biệt nào đó.

    `seed` cố định để chạy lại cho ra đúng cùng tập nhiễu, phục vụ tái lập.
    `source` ghi lại nguồn câu để benchmark tách được hiệu năng theo từng nguồn.
    """
    gen = noise_gen or ASRNoiseGenerator(profile=profile, seed=seed)
    rows = []
    for i, s in enumerate(sentences):
        noisy, errs = gen.apply_noise(s)
        rows.append({"id": i + 1, "noisy": noisy, "clean": s,
                     "errors": errs, "split": tag, "source": source})
    return rows


def build_identity_test(sentences: Sequence[str], source: str = "unknown") -> List[Dict]:
    """Tập đối chứng: đầu vào = đầu ra = câu đã đúng.

    Đây là phép đo duy nhất bắt được over-correction thuần. Nếu mô hình sửa bất cứ gì
    trên tập này thì phần sửa đó chắc chắn sai.
    """
    return [
        {"id": i + 1, "noisy": s, "clean": s, "errors": ["clean"],
         "split": "identity", "source": source}
        for i, s in enumerate(sentences)
    ]


def format_stats(stats: Dict[str, int]) -> str:
    """In thống kê dọn dữ liệu thành một khối dễ đọc, để dán vào log chạy."""
    lines = [
        "─── Kiểm toán dữ liệu ───────────────────────────",
        f"  Câu đưa vào          : {stats.get('input', 0):>10,}",
        f"  Loại TRÙNG LẶP        : {stats.get('duplicate', 0):>10,}",
        f"  Loại câu RÁC          : {stats.get('rác', 0):>10,}",
        f"  Loại quá ngắn/dài     : {stats.get('quá_ngắn', 0) + stats.get('quá_dài', 0):>10,}",
    ]
    if stats.get("cắt_bớt"):
        lines.append(f"  Cắt bớt (đủ chỉ tiêu) : {stats['cắt_bớt']:>10,}")
    lines += [
        f"  Còn lại               : {stats.get('giữ_lại', 0):>10,}",
        "  ─────────────────────────────────────────────",
        f"  Train                 : {stats.get('train', 0):>10,}",
        f"  Validation            : {stats.get('validation', 0):>10,}",
        f"  Trùng train↔val       : {stats.get('trùng_train_val', 0):>10,}"
        f"{'  <-- PHẢI LÀ 0' if stats.get('trùng_train_val', 0) == 0 else '  <-- LỖI!'}",
        "─────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


def write_jsonl(rows: Sequence[Dict], path) -> int:
    """Ghi danh sách cặp câu ra .jsonl, trả về số dòng đã ghi."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
