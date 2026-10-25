from src.baseline.rule_baseline import RuleAndDictionaryBaseline
from src.srt_utils.srt_processor import SRTSubtitleProcessor


def test_srt_processor_preserves_timestamps():
    sample_srt = """1
00:00:01,000 --> 00:00:03,500
toi di lam o ngan hang va gianh tien tiet kiem

2
00:00:04,000 --> 00:00:07,000
hom nay thoi tiet rat dep
"""
    baseline = RuleAndDictionaryBaseline()
    processor = SRTSubtitleProcessor(corrector_model=baseline)

    output_srt = processor.process_srt_string(sample_srt)

    # Kiểm tra timestamp được giữ nguyên trong văn bản SRT
    assert "00:00:01,000 --> 00:00:03,500" in output_srt
    assert "00:00:04,000 --> 00:00:07,000" in output_srt
    # Kiểm tra nội dung đã được sửa
    assert "dành tiền" in output_srt


def test_srt_processor_empty():
    processor = SRTSubtitleProcessor()
    assert processor.process_srt_string("") == ""
