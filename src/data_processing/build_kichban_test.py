"""Dựng tập test từ phụ đề ASR thật (kichban_*.srt) + bản gán nhãn của người kiểm.

Vì sao cần file này: `data/gold_test/sample_test.jsonl` (25 câu) có `noisy` là câu KHÔNG DẤU,
nhưng ASR thật (Whisper) xuất ra văn bản CÓ DẤU THANH 99,3%, CÓ dấu câu 81,1%. Tập test cũ
đo một tác vụ không tồn tại. File này thay bằng dữ liệu thật.

Cấu trúc gán nhãn:
    FIXES           câu nào Whisper sai -> bản đúng. Câu không có trong FIXES nghĩa là
                    đã đúng hoàn toàn, `clean` = `noisy` (mẫu identity tự nhiên).
    LOW_CONFIDENCE  chỗ người gán nhãn KHÔNG CHẮC (không nghe được audio, chỉ suy từ ngữ
                    cảnh). Bắt buộc nghe lại video trước khi chốt.
    EXCLUDE         block bị loại khỏi tập test (quảng cáo sản phẩm người lớn, block 133-148).

Chạy:
    python -m src.data_processing.build_kichban_test
"""

import argparse
import re
import sys
from pathlib import Path

# ── Bản gán nhãn ──────────────────────────────────────────────────────────────
# Chỉ liệt kê câu CẦN SỬA. Câu không có ở đây được coi là đã đúng.
FIXES = {
    1: "Xin chào tất cả quý vị và các bạn cùng đến với chương trình Tổ Buôn 247.",
    2: "Ngày hôm nay thì câu chuyện khóa học làm giàu có giá lên tới 1,4 tỷ,",
    5: "thì không hiểu là dạy kinh doanh rồi thì tư vấn về kinh doanh kiểu gì, chắc cũng là loại lùa gà thôi.",
    15: "nhằm củng cố cho việc họ đang lùa gà thì để mà nói nó cũng rất là ôi rồi ôi.",
    16: "Cái chuyện mà diễn giả mặt búng ra sữa hay là diễn giả mặt rất là như đi dạy học vậy thì nó cũng chẳng nói lên điều gì cả.",
    18: "Chứ họ có kinh nghiệm gì để chuyên tâm dạy lại đâu.",
    19: "Thế nhưng mà ngược lại, rất nhiều người trẻ họ lại đang thành công rực rỡ,",
    20: "họ điều hành đủ các loại doanh nghiệp thì kiêm việc mà họ đứng ở trên bục họ dạy dỗ người khác,",
    22: "Biết bao nhiêu tỉ phú ở trên thế giới hiện tại họ đều là những người trẻ mà đấy thôi,",
    23: "Bill Gates hay là Mark Zuckerberg, 19 tuổi đã thành công rồi mà.",
    24: "Nếu mà học được những cái hay từ những con người đó để áp dụng vào cuộc sống của mình thì nó quá tốt đi ấy chứ.",
    27: "nó mới là cái đội siêu siêu giỏi luôn,",
    28: "nó đẻ ra một cái thôi, nó đã biết bơi trước bố nó rồi.",
    31: "Không tin anh em cứ ngẫm mà xem, người thành công cần phải có cái tính tò mò đúng không?",
    32: "Cái gì cũng cần phải đặt câu hỏi, kể cả là những câu hỏi ngớ ngẩn nhất.",
    33: "Vậy thì thử hỏi, có ai tò mò hơn bọn trẻ con không?",
    37: "Nó đã đòi cái gì ấy thì nó khóc, nó quấy, nó ăn vạ, nó làm đủ mọi cách để đạt được thì thôi.",
    38: "Người lớn hứa với nó cái gì mà không làm ấy thì chết với chúng nó luôn, nó nhớ dai vô cùng nhé.",
    43: "Thế nhưng mà thử hỏi có người lớn nào thích nghi nhanh được như bọn trẻ con hay không nào?",
    44: "Ê, vứt vào một môi trường mới nhé.",
    47: "Vậy thì cái việc một đứa trẻ nó đứng lên, nó làm diễn giả, nó dạy dỗ cho người ở dưới.",
    52: "Đấy mới là cái đáng để xoáy vào nhất.",
    55: "Thế nhưng năng lực thực tế của họ như thế nào ấy thì đấy mới là cái đáng để xét.",
    56: "Giới thiệu diễn giả là người từng làm chức nọ kia trước kia ở Hugo.",
    60: "đến các diễn giả dạy làm giàu.",
    61: "Thế nhưng mà công ty của họ lại lỗ chỏng vó ra đây thôi.",
    62: "Thế thôi thì ở trên thế giới tác giả của cuốn Cha giàu Cha nghèo hoành tráng ấy.",
    63: "Nhưng mà công ty của ông ấy cũng mấy lần xin phá sản rồi còn gì nữa.",
    66: "Thế nhưng có một cái điều chắc chắn bạn sẽ không bao giờ nhận được ở các cái lớp dạy làm giàu.",
    75: "Nhưng mà bây giờ tôi ở một túp lều tranh nhưng mà tôi cảm thấy hạnh phúc, tôi cảm thấy vui vẻ.",
    76: "Vậy là thành công rồi còn hơn là ở trong biệt thự.",
    78: "Người thì lo là sau này chia chác tài sản như nào cho công bằng đau đầu lắm.",
    83: "Vậy thì thử hỏi thằng có nguồn lực một nhưng nó phấn đấu lên được năm.",
    86: "Ví dụ như kiểu cái việc là ông Donald Trump ông ấy sinh ra trong một gia đình quý tộc.",
    91: "Ngay từ cái định nghĩa về thành công của mỗi người nó đã khác nhau rồi.",
    94: "Anh dạy cho mày cách kiếm 500 triệu thì nghe nó còn có lý.",
    96: "Để ông ấy dạy cho công thức thành công thì ở đấy chắc chắn là lùa gà thôi.",
    97: "Ngay cả khi hai bên đều có chung một cái định nghĩa rằng thành công là nhiều tiền.",
    98: "Thì cái việc dạy cho người khác kiếm thật nhiều tiền",
    101: "Bill Gates thì có gì có thể giàu được.",
    103: "và cái thế đứng của ông ấy thì nó nằm ở cái thế đó gọi là tọa sơn hướng thủy rồi.",
    104: "Ông bà già của Bill Gates là cái dân sừng sỏ về luật và ngân hàng",
    105: "quan hệ của họ cực kỳ rộng, nhà của họ bố của giàu luôn rồi.",
    106: "Trong khi đấy ông Bill Gates, ông ấy lại còn là cái dạng đầu có sạn nữa chứ.",
    107: "Đầu của Bill Gates cắm sạc cho xe Tesla thì lại chạy vòng quanh nước Mỹ cũng được luôn.",
    108: "Vậy thì cái việc ông ấy trở nên giàu quá cũng là điều dễ hiểu thôi mà",
    109: "Chứ nếu như bây giờ bảo rằng Bill Gates sinh ra vào cái thời bây giờ",
    112: "Bố mẹ làm nông rồi chạy ăn từng bữa.",
    113: "Vậy thì thử hỏi lấy đâu ra mà đòi làm tỷ phú thế giới.",
    114: "Mơ lên đến được cái chức trưởng phòng thôi đấy là hơn năm rồi.",
    115: "Chính vì thế nên cái việc đọc tiểu sự của Bill Gates để học cách làm giàu",
    117: "Nếu có thì chỉ có học được cái ý chí, học được nghị lực, học được cái niềm đam mê của ông ấy thôi.",
    120: "Đấy có cái đấy là đáng học hỏi được từ Bill Gates thôi.",
    121: "Chứ còn bây giờ đòi gặp Bill Gates để xin công thức thành công.",
    122: "Thế mà làm thế nào để được giàu như anh hả Bill Gates?",
    123: "Ấy thì Bill Gates, chính Bill Gates cũng chịu luôn.",
    130: "Thế nhưng mà cái công thức đấy thì giá của nó rẻ lắm.",
    132: "Anh em chỉ cần ấn vào đường link phía dưới phần bình luận là thấy luôn.",
    118: "Nhà ông ấy nghèo khó đổ vách rồi mà ông ấy vẫn khổ quốc như châu, ông ấy làm ngày làm đêm, làm quên ăn quên ngủ.",
    119: "À trong khi đấy mình đây lương có mười mấy triệu nhưng mình suốt ngày đòi chữa lành kia.",
}

# Chỗ người gán nhãn KHÔNG CHẮC — phải nghe lại audio mới chốt được.
LOW_CONFIDENCE = {18, 24, 27, 28, 43, 52, 56, 61, 62, 101, 103, 104, 105, 106, 107, 113, 114, 118, 119}

# Block bị loại: quảng cáo sản phẩm người lớn, nội dung nhạy cảm + quá rối để gán nhãn.
EXCLUDE = set(range(133, 149))


def parse_srt(text: str):
    """Bóc file SRT thành danh sách (index, start, end, nội dung)."""
    pattern = re.compile(
        r"(\d+)\s*\n(\d\d:\d\d:\d\d,\d+)\s*-->\s*(\d\d:\d\d:\d\d,\d+)\s*\n([\s\S]*?)(?=\n\d+\s*\n\d\d:|\Z)"
    )
    out = []
    for m in pattern.finditer(text):
        out.append({
            "index": int(m.group(1)),
            "start": m.group(2),
            "end": m.group(3),
            "noisy": " ".join(m.group(4).split()),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Dựng tập test từ phụ đề ASR thật")
    parser.add_argument("--srt", default="data/raw/kichban_2026-10-25.srt")
    parser.add_argument("--out_dir", default="data/gold_test/gd2_real_asr")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.exists():
        print(f"[LỖI] Không thấy {srt_path}")
        return 1

    blocks = parse_srt(srt_path.read_text(encoding="utf-8", errors="replace"))
    print(f"Đọc {srt_path.name}: {len(blocks)} block")

    rows, n_fixed, n_clean = [], 0, 0
    for b in blocks:
        if b["index"] in EXCLUDE:
            continue
        clean = FIXES.get(b["index"], b["noisy"])
        if clean != b["noisy"]:
            n_fixed += 1
        else:
            n_clean += 1
        rows.append({
            "id": b["index"],
            "noisy": b["noisy"],
            "clean": clean,
            "error_type": ["asr_real"] + (["low_confidence"] if b["index"] in LOW_CONFIDENCE else []),
            "start": b["start"],
            "end": b["end"],
            "source": "whisper_real_kichban",
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import json

    jsonl_path = out_dir / "kichban_test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Bản đọc để NGƯỜI duyệt — chỉ hiện câu có sửa, để tập trung vào chỗ cần kiểm
    md = ["# Duyệt bản gán nhãn — kichban_2026-10-25.srt", "",
          f"- Tổng block giữ lại: **{len(rows)}** (bỏ {len(EXCLUDE)} block quảng cáo)",
          f"- Câu cần sửa: **{n_fixed}** · câu đã đúng, giữ nguyên: **{n_clean}**",
          f"- ⚠️ cần nghe lại audio: **{len(LOW_CONFIDENCE)}** câu", "",
          "| # | ⚠️ | Whisper (nguyên văn) | Đề xuất (bản đúng) |",
          "| ---: | :---: | :--- | :--- |"]
    for r in rows:
        if r["noisy"] == r["clean"]:
            continue
        flag = "⚠️" if r["id"] in LOW_CONFIDENCE else ""
        md.append(f'| {r["id"]} | {flag} | {r["noisy"]} | **{r["clean"]}** |')

    md_path = out_dir / "kichban_review.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nGhi {jsonl_path}  ({len(rows)} câu)")
    print(f"Ghi {md_path}  (bảng duyệt, chỉ hiện {n_fixed} câu có sửa)")
    print(f"\n  câu đã đúng, giữ nguyên : {n_clean}")
    print(f"  câu cần sửa             : {n_fixed}")
    print(f"  ⚠️  cần nghe lại audio  : {len(LOW_CONFIDENCE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
