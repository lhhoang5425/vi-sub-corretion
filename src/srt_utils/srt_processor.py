"""Mô-đun bóc tách và xử lý tệp phụ đề .SRT (SRT Subtitle Processor)

Chức năng:
1. Parse tệp .srt thành các khối subtitle.
2. Trích xuất danh sách câu thô (noisy text).
3. Đưa qua mô hình sửa lỗi theo batch (Seq2Seq hoặc Baseline).
4. Ghép câu đã chuẩn hóa vào đúng timestamp ban đầu.
5. Xuất lại file .srt hoàn chỉnh.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import re
from typing import List, Optional, Union

try:
    import srt
except ImportError:
    srt = None


@dataclass
class FallbackSubtitleItem:
    index: int
    start: timedelta
    end: timedelta
    content: str


def _parse_srt_timestamp(ts: str) -> timedelta:
    # 00:00:01,000
    parts = ts.strip().replace(",", ":").split(":")
    hours = int(parts[0])
    mins = int(parts[1])
    secs = int(parts[2])
    millis = int(parts[3])
    return timedelta(hours=hours, minutes=mins, seconds=secs, milliseconds=millis)


def _format_srt_timestamp(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


class SRTSubtitleProcessor:
    def __init__(self, corrector_model=None):
        """corrector_model: Object có hàm .predict_batch(List[str]) -> List[str]"""
        self.corrector_model = corrector_model

    def _parse_fallback(self, srt_content: str) -> List[FallbackSubtitleItem]:
        blocks = re.split(r"\n\s*\n", srt_content.strip())
        items = []
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) >= 3 and "-->" in lines[1]:
                idx = int(lines[0])
                time_parts = lines[1].split("-->")
                start = _parse_srt_timestamp(time_parts[0])
                end = _parse_srt_timestamp(time_parts[1])
                text = " ".join(lines[2:])
                items.append(FallbackSubtitleItem(index=idx, start=start, end=end, content=text))
        return items

    def _compose_fallback(self, items: List[FallbackSubtitleItem]) -> str:
        blocks = []
        for item in items:
            t_start = _format_srt_timestamp(item.start)
            t_end = _format_srt_timestamp(item.end)
            blocks.append(f"{item.index}\n{t_start} --> {t_end}\n{item.content}")
        return "\n\n".join(blocks) + "\n"

    def process_srt_string(self, srt_content: str, batch_size: int = 16) -> str:
        """Nhận vào nội dung xâu .srt và trả về xâu .srt đã được chuẩn hóa."""
        if not srt_content.strip():
            return ""

        # 1. Parse các block phụ đề
        if srt is not None:
            subtitles = list(srt.parse(srt_content))
            if not subtitles:
                return srt_content
            raw_texts = [sub.content.replace("\n", " ").strip() for sub in subtitles]
        else:
            subtitles = self._parse_fallback(srt_content)
            if not subtitles:
                return srt_content
            raw_texts = [sub.content for sub in subtitles]

        # 2. Chạy qua mô hình sửa lỗi
        if self.corrector_model is not None:
            cleaned_texts = self.corrector_model.predict_batch(raw_texts, batch_size=batch_size)
        else:
            cleaned_texts = raw_texts

        # 3. Gán câu sạch vào lại từng block và giữ nguyên timestamp
        for sub, clean_text in zip(subtitles, cleaned_texts):
            sub.content = clean_text

        # 4. Soạn thảo lại định dạng chuẩn SRT
        if srt is not None:
            return srt.compose(subtitles)
        else:
            return self._compose_fallback(subtitles)

    def srt_to_txt(self, srt_content: str) -> str:
        """Trích xuất văn bản thuần túy từ tệp phụ đề .srt (loại bỏ timestamps và số thứ tự).
        
        Rất hữu ích để lấy dữ liệu làm sạch nạp vào tập dữ liệu huấn luyện cho mô hình.
        """
        if not srt_content.strip():
            return ""

        if srt is not None:
            subtitles = list(srt.parse(srt_content))
            lines = [sub.content.replace("\n", " ").strip() for sub in subtitles]
        else:
            subtitles = self._parse_fallback(srt_content)
            lines = [sub.content for sub in subtitles]

        return "\n".join(line for line in lines if line)

    def process_file_flexible(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
    ) -> dict:
        """Xử lý linh hoạt cả tệp .srt và .txt:
        - Nếu đầu vào là .srt: Sửa lỗi và xuất ra cả file .srt chuẩn (giữ timestamps) và file .txt sạch.
        - Nếu đầu vào là .txt: Sửa lỗi và xuất ra file .txt sạch.
        """
        in_path = Path(input_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {in_path}")

        target_dir = Path(output_dir) if output_dir else in_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if in_path.suffix.lower() == ".srt":
            # 1. Xuất file .srt sạch (giữ timestamps)
            cleaned_srt = self.process_srt_string(content, batch_size=batch_size)
            out_srt = target_dir / f"{in_path.stem}_cleaned.srt"
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(cleaned_srt)
            results["cleaned_srt"] = str(out_srt)

            # 2. Xuất thêm file .txt thuần túy sạch (không timestamps)
            cleaned_txt = self.srt_to_txt(cleaned_srt)
            out_txt = target_dir / f"{in_path.stem}_cleaned.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(cleaned_txt)
            results["cleaned_txt"] = str(out_txt)

        elif in_path.suffix.lower() == ".txt":
            # Đầu vào là file text thuần: sửa từng dòng
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            if self.corrector_model is not None:
                cleaned_lines = self.corrector_model.predict_batch(lines, batch_size=batch_size)
            else:
                cleaned_lines = lines

            out_txt = target_dir / f"{in_path.stem}_cleaned.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(cleaned_lines) + "\n")
            results["cleaned_txt"] = str(out_txt)
        else:
            raise ValueError("Chỉ hỗ trợ tệp định dạng .srt hoặc .txt")

        return results

    def process_srt_file(
        self,
        input_srt_path: Union[str, Path],
        output_srt_path: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
    ) -> Path:
        """Đọc file .srt từ ổ đĩa, sửa lỗi và lưu ra file mới."""
        in_path = Path(input_srt_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {in_path}")

        if output_srt_path is None:
            out_path = in_path.with_name(f"{in_path.stem}_cleaned.srt")
        else:
            out_path = Path(output_srt_path)

        with open(in_path, "r", encoding="utf-8") as f:
            content = f.read()

        cleaned_content = self.process_srt_string(content, batch_size=batch_size)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)

        print(f"[SRTProcessor] Đã lưu phụ đề đã sửa tại: {out_path}")
        return out_path


if __name__ == "__main__":
    # Test mẫu
    sample_srt = """1
00:00:01,000 --> 00:00:03,500
xin chao cac ban da den voi video

2
00:00:04,000 --> 00:00:07,000
hom nay toi se huong dan cac ban hoc may
"""
    from ..baseline.rule_baseline import RuleAndDictionaryBaseline

    dummy_baseline = RuleAndDictionaryBaseline()
    processor = SRTSubtitleProcessor(corrector_model=dummy_baseline)

    result = processor.process_srt_string(sample_srt)
    print("=== KẾT QUẢ SRT SAU KHI SỬA ===")
    print(result)
