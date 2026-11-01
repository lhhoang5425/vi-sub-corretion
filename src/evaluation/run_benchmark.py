"""Script chạy đánh giá đối đầu Benchmark (Evaluation Benchmark Runner)

Thực hiện đo đạc định lượng trực tiếp để hoàn thiện Mục 4 & Mục 6 trong Đề cương NLP-28:
1. Baseline 1: Identity (giữ nguyên câu ASR lỗi)
2. Baseline 2: Rule-based & Dictionary (tra từ điển và luật viết hoa/chấm câu)
3. Seq2Seq: Mô hình Deep Learning Transformer (bmd1905 / BARTpho fine-tuned)

Tính toán các chỉ số:
- SacreBLEU / BLEU (Độ tương đồng câu với Ground Truth)
- WER (%) (Word Error Rate - Tỷ lệ lỗi ở mức từ)
- Edit-level Precision, Recall, F1 (Đo lường độ chính xác tại các vị trí sửa lỗi)

Hỗ trợ xuất báo cáo:
- Tệp Markdown (.md)
- Tệp Excel (.xlsx) với 2 Sheet: Sheet Tổng Hợp Chỉ Số và Sheet Chi Tiết Từng Câu
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..baseline.rule_baseline import IdentityBaseline, RuleAndDictionaryBaseline
from .metrics import evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy kiểm thử Benchmark cho Đề tài NLP-28")
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/gold_test/sample_test.jsonl",
        help="Đường dẫn tệp kiểm thử chuẩn (JSONL)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bmd1905/vietnamese-correction-v2",
        help="Đường dẫn checkpoint hoặc tên mô hình HuggingFace",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="reports/benchmark_summary.md",
        help="Tệp lưu báo cáo kết quả Markdown",
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default="reports/benchmark_results.xlsx",
        help="Tệp lưu báo cáo kết quả Excel (.xlsx)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Kích thước batch khi suy luận",
    )
    return parser.parse_args()


def load_test_data(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file kiểm thử: {path}")

    sources = []
    references = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sources.append(data["noisy"])
            references.append(data["clean"])

    return sources, references


def format_markdown_table(results: Dict[str, Dict[str, float]], num_samples: int) -> str:
    md = []
    md.append(f"### Bảng kết quả đánh giá Benchmark định lượng (Tập Test: {num_samples} mẫu)\n")
    md.append("| Phương pháp đối chứng | BLEU ↑ | WER (%) ↓ | Edit-P (%) ↑ | Edit-R (%) ↑ | Edit-F1 (%) ↑ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for name, m in results.items():
        md.append(
            f"| **{name}** | **{m['bleu']:.2f}** | **{m['wer']:.2f}%** | "
            f"{m['edit_precision']:.2f}% | {m['edit_recall']:.2f}% | **{m['edit_f1']:.2f}%** |"
        )
    md.append("\n*Ghi chú:*")
    md.append("- **BLEU ↑**: Càng cao càng tốt (Đo độ trùng khớp ngữ nghĩa n-gram với câu chuẩn).")
    md.append("- **WER ↓**: Càng thấp càng tốt (Tỷ lệ từ sai khác so với câu chuẩn).")
    md.append("- **Edit-F1 ↑**: Đo lường sự chính xác tại các vị trí xảy ra phép sửa lỗi (tránh sửa nhầm câu đúng).")
    return "\n".join(md)


def export_to_excel(
    results: Dict[str, Dict[str, float]],
    sources: List[str],
    references: List[str],
    predictions: Dict[str, List[str]],
    output_excel_path: str,
):
    """Xuất báo cáo benchmark ra tệp Excel (.xlsx) gồm 2 Sheet trực quan:
    1. 'TongHop_ChiSo': Bảng tổng hợp các chỉ số BLEU, WER, Edit-F1.
    2. 'ChiTiet_TungCau': Bảng chi tiết từng câu đầu vào, câu chuẩn và câu sửa lỗi của từng mô hình.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[Cảnh báo] Chưa cài đặt pandas, không thể xuất file Excel.")
        return

    out_path = Path(output_excel_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Sheet Tổng hợp
    summary_rows = []
    for model_name, metrics in results.items():
        summary_rows.append({
            "Phương pháp đối chứng": model_name,
            "BLEU (Độ tương đồng) ↑": metrics["bleu"],
            "WER (Tỷ lệ lỗi từ %) ↓": metrics["wer"],
            "Edit Precision (%) ↑": metrics["edit_precision"],
            "Edit Recall (%) ↑": metrics["edit_recall"],
            "Edit F1 (%) ↑": metrics["edit_f1"],
        })
    df_summary = pd.DataFrame(summary_rows)

    # 2. Sheet Chi tiết từng câu
    detail_rows = []
    for idx, (src, ref) in enumerate(zip(sources, references)):
        row = {
            "STT": idx + 1,
            "Câu gốc ASR (Noisy)": src,
            "Đáp án chuẩn (Ground Truth)": ref,
        }
        for model_name, preds in predictions.items():
            row[f"Kết quả {model_name}"] = preds[idx] if idx < len(preds) else ""
        detail_rows.append(row)
    df_detail = pd.DataFrame(detail_rows)

    # Ghi ra file Excel định dạng chuẩn
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="TongHop_ChiSo", index=False)
            df_detail.to_excel(writer, sheet_name="ChiTiet_TungCau", index=False)
        print(f"[Thông báo] Đã xuất báo cáo Excel thành công tại: {out_path}")
    except Exception as e:
        csv_path = out_path.with_suffix(".csv")
        df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[Thông báo] Đã lưu dự phòng dạng CSV tại: {csv_path} (Lỗi openpyxl: {e})")


def run_benchmark():
    args = parse_args()
    print("=" * 65)
    print("BẮT ĐẦU CHẠY ĐÁNH GIÁ BENCHMARK ĐỐI ĐẦU CHO ĐỀ TÀI NLP-28")
    print("=" * 65)
    print(f"- Tập dữ liệu test : {args.test_file}")
    print(f"- Mô hình Seq2Seq  : {args.model_name_or_path}")

    sources, references = load_test_data(args.test_file)
    print(f"- Số lượng mẫu test: {len(sources)} câu\n")

    all_results = {}
    all_predictions = {}

    # 1. Baseline 1: Identity
    print("[1/3] Đang đánh giá Baseline 1: Identity (giữ nguyên)...")
    baseline1 = IdentityBaseline()
    preds_b1 = baseline1.predict_batch(sources)
    res_b1 = evaluate_predictions(sources, preds_b1, references)
    all_results["Baseline 1 (Identity)"] = res_b1
    all_predictions["Baseline 1 (Identity)"] = preds_b1
    print(f"      -> BLEU: {res_b1['bleu']} | WER: {res_b1['wer']}% | Edit-F1: {res_b1['edit_f1']}%")

    # 2. Baseline 2: Rule & Dictionary
    print("[2/3] Đang đánh giá Baseline 2: Rule & Dictionary...")
    baseline2 = RuleAndDictionaryBaseline()
    preds_b2 = baseline2.predict_batch(sources)
    res_b2 = evaluate_predictions(sources, preds_b2, references)
    all_results["Baseline 2 (Rule & Dictionary)"] = res_b2
    all_predictions["Baseline 2 (Rule & Dictionary)"] = preds_b2
    print(f"      -> BLEU: {res_b2['bleu']} | WER: {res_b2['wer']}% | Edit-F1: {res_b2['edit_f1']}%")

    # 3. Mô hình Seq2Seq
    seq2seq_name = f"Seq2Seq ({Path(args.model_name_or_path).name})"
    print(f"[3/3] Đang đánh giá Seq2Seq ({args.model_name_or_path})...")
    from ..models.inference import Seq2SeqCorrector
    seq2seq = Seq2SeqCorrector(model_name_or_path=args.model_name_or_path)
    preds_seq2seq = seq2seq.predict_batch(sources, batch_size=args.batch_size)
    res_seq2seq = evaluate_predictions(sources, preds_seq2seq, references)
    all_results[seq2seq_name] = res_seq2seq
    all_predictions[seq2seq_name] = preds_seq2seq
    print(f"      -> BLEU: {res_seq2seq['bleu']} | WER: {res_seq2seq['wer']}% | Edit-F1: {res_seq2seq['edit_f1']}%")

    # Xuất bảng Markdown
    table_md = format_markdown_table(all_results, len(sources))
    print("\n" + "=" * 65)
    print(table_md)
    print("=" * 65)

    # 1. Lưu báo cáo Markdown
    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table_md + "\n")
        print(f"\n[Thông báo] Đã lưu bảng kết quả Markdown tại: {out_path}")

    # 2. Xuất báo cáo Excel (nhiều sheet)
    if args.output_excel:
        export_to_excel(
            results=all_results,
            sources=sources,
            references=references,
            predictions=all_predictions,
            output_excel_path=args.output_excel,
        )

    return all_results


if __name__ == "__main__":
    run_benchmark()
