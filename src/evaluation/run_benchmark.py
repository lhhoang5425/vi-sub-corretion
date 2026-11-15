"""Script chạy đánh giá đối đầu Benchmark (Evaluation Benchmark Runner)

Đo đạc định lượng cho Mục 4 & Mục 6 Đề cương NLP-28, đối đầu 3 phương pháp:
1. Baseline 1: Identity (giữ nguyên câu ASR lỗi)
2. Baseline 2: Rule-based & Dictionary (tra từ điển và luật viết hoa/chấm câu)
3. Seq2Seq: Mô hình Transformer fine-tune (bmd1905 / BARTpho / ViT5)

Chỉ số: BLEU, WER, CER, Exact Match, ROUGE-L, Edit-level P/R/F1.

Ba định dạng đầu ra, mỗi loại cho một mục đích:
- .md   : bảng tổng hợp dán thẳng vào đề cương / báo cáo
- .txt  : log đầy đủ chạy trên Colab, gồm cấu hình + bảng + từng câu + phân tích lỗi
- .xlsx : hai sheet Tổng Hợp Chỉ Số và Chi Tiết Từng Câu để mở bằng Excel
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from ..baseline.rule_baseline import IdentityBaseline, RuleAndDictionaryBaseline
from .metrics import evaluate_predictions

METRIC_COLUMNS = [
    ("bleu", "BLEU ↑"),
    ("wer", "WER (%) ↓"),
    ("cer", "CER (%) ↓"),
    ("exact_match", "Exact Match (%) ↑"),
    ("rouge_l", "ROUGE-L (%) ↑"),
    ("edit_precision", "Edit-P (%) ↑"),
    ("edit_recall", "Edit-R (%) ↑"),
    ("edit_f1", "Edit-F1 (%) ↑"),
]


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
        help="Tệp lưu báo cáo Markdown (bảng tổng hợp)",
    )
    parser.add_argument(
        "--output_log",
        type=str,
        default="reports/benchmark_log.txt",
        help="Tệp lưu log đánh giá ĐẦY ĐỦ dạng text: cấu hình, bảng, từng câu, phân tích lỗi",
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default="reports/benchmark_results.xlsx",
        help="Tệp lưu báo cáo Excel (.xlsx)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Kích thước batch khi suy luận",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Tệp JSON chứa RAW metrics của mọi phương pháp + cấu hình giải mã. "
             "compare_runs.py đọc các tệp này để đặt cạnh nhau nhiều đợt train.",
    )
    parser.add_argument("--num_beams", type=int, default=5, help="Số beam khi giải mã")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Phạt lặp. 1.2 đang là mặc định và KHÁ Tay; grid-search 1.0-1.15")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Thiên vị độ dài khi beam search")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Cấm lặp n-gram")
    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Cho phép rơi về Rule Baseline nếu nạp mô hình thất bại. MẶC ĐỊNH TẮT: "
             "thiếu checkpoint thì dừng ngay, không đo nhầm baseline rồi báo là Seq2Seq.",
    )
    return parser.parse_args()


def load_test_data(file_path: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """Đọc tập kiểm thử, trả về (noisy, clean, nhãn lỗi theo từng câu).

    Trường error_type trong gold test vốn đã được gán nhãn thủ công; dùng nó để tách
    bảng hiệu năng theo từng loại lỗi (mất dấu, mất dấu câu, từ đồng âm...) thay vì
    chỉ ra một con số trung bình chung.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file kiểm thử: {path}")

    sources, references, error_types = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sources.append(data["noisy"])
            references.append(data["clean"])
            error_types.append(data.get("error_type") or ["khong_quy_nhan"])

    return sources, references, error_types


def format_markdown_table(results: Dict[str, Dict[str, float]], num_samples: int) -> str:
    """Bảng đối đầu chính: mỗi phương pháp một hàng, mỗi chỉ số một cột."""
    head = "| Phương pháp | " + " | ".join(label for _, label in METRIC_COLUMNS) + " |"
    sep = "| :--- |" + " | ".join(":---:" for _ in METRIC_COLUMNS) + " |"

    md = [f"### Bảng kết quả Benchmark định lượng (Tập Test: {num_samples} mẫu)\n", head, sep]
    for name, m in results.items():
        cells = []
        for key, _ in METRIC_COLUMNS:
            # BLEU là điểm thang 0-100 theo quy ước, không viết dưới dạng phần trăm
            unit = "" if key == "bleu" else "%"
            bold = key in ("bleu", "wer", "exact_match", "edit_f1")
            text = f"{m[key]:.2f}{unit}"
            cells.append(f"**{text}**" if bold else text)
        md.append(f"| **{name}** | " + " | ".join(cells) + " |")

    md += [
        "\n*Ghi chú chỉ số:*",
        "- **BLEU / ROUGE-L ↑**: độ tương đồng với câu chuẩn (BLEU thiên về n-gram khớp hẳn, ROUGE-L thiên về bao phủ).",
        "- **WER ↓ / CER ↓**: tỷ lệ lệch ở mức từ và mức KÝ TỰ. CER mới thấy được lỗi thiếu dấu thanh,",
        "  vì WER vẫn đếm một từ sai dấu là đúng 1 từ sai dù chỉ lệch một ký tự.",
        "- **Exact Match ↑**: tỷ lệ câu trùng 100% đáp án chuẩn. Con số dễ đọc nhất cho hội đồng.",
        "- **Edit-P ↑**: trong các câu mô hình đụng vào sửa, bao nhiêu % sửa đúng đích (càng thấp càng over-correct).",
        "- **Edit-R ↑**: bao nhiêu % câu cần sửa thì được can thiệp (100% = không bỏ sót câu nào).",
        "- **Edit-F1 ↑**: trung bình điều hòa của Edit-P và Edit-R.",
    ]
    return "\n".join(md)


def format_error_type_table(
    predictions: Dict[str, List[str]],
    sources: List[str],
    references: List[str],
    error_types: List[List[str]],
) -> str:
    """Bảng hiệu năng TÁCH THEO loại lỗi, chỉ dành cho mô hình Seq2Seq.

    Trả lời câu hỏi 'mô hình mạnh ở đâu và yếu ở đâu' mà bảng trung bình che mất:
    ví dụ phục hồi dấu thanh tốt nhưng khôi phục dấu câu kém.
    """
    seq2seq_names = [n for n in predictions if n.startswith("Seq2Seq")]
    if not seq2seq_names:
        return ""
    model_name = seq2seq_names[0]
    preds = predictions[model_name]

    buckets: Dict[str, List[int]] = {}
    for idx, types in enumerate(error_types):
        for t in types:
            buckets.setdefault(t, []).append(idx)

    rows = []
    for bucket, indices in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        sub_src = [sources[i] for i in indices]
        sub_pred = [preds[i] for i in indices]
        sub_ref = [references[i] for i in indices]
        m = evaluate_predictions(sub_src, sub_pred, sub_ref)
        rows.append((bucket, len(indices), m["exact_match"], m["cer"], m["bleu"], m["edit_f1"]))

    md = [
        f"\n### Bảng hiệu năng theo loại lỗi ({model_name})\n",
        "| Loại lỗi | Số câu | Exact Match (%) ↑ | CER (%) ↓ | BLEU ↑ | Edit-F1 (%) ↑ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for bucket, n, em, cer, bleu, f1 in rows:
        md.append(f"| {bucket} | {n} | {em:.2f} | {cer:.2f} | {bleu:.2f} | {f1:.2f} |")
    md.append("\n*Một câu mang nhiều nhãn lỗi (vd: tones + punctuation) được đếm ở cả hai dòng,*")
    md.append("*nên tổng Số câu có thể lớn hơn số mẫu của tập test.*")
    return "\n".join(md)


def format_case_review(
    predictions: Dict[str, List[str]],
    sources: List[str],
    references: List[str],
    error_types: List[List[str]],
    max_cases: int = 0,
) -> str:
    """Liệt kê từng câu: gốc ASR / đáp án chuẩn / kết quả mô hình, kèm phân loại kết quả."""
    seq2seq_names = [n for n in predictions if n.startswith("Seq2Seq")]
    model_name = seq2seq_names[0] if seq2seq_names else list(predictions)[-1]
    preds = predictions[model_name]

    lines = [f"\n{'=' * 78}", f"PHÂN TÍCH TỪNG CÂU ({model_name})", "=" * 78]
    picked = range(len(sources)) if max_cases <= 0 else range(min(max_cases, len(sources)))

    counts = {"DUNG": 0, "GAN DUNG": 0, "SAI": 0}
    for i in picked:
        src, pred, ref = sources[i], preds[i], references[i]
        if pred.strip() == ref.strip():
            verdict = "DUNG"
        elif pred.strip() == src.strip():
            verdict = "SAI (giu nguyen)"
            counts["SAI"] += 1
        else:
            verdict = "GAN DUNG"
            counts["GAN DUNG"] += 1
        if verdict == "DUNG":
            counts["DUNG"] += 1

        lines += [
            f"\n[{i + 1}] ({', '.join(error_types[i])})  ket qua: {verdict}",
            f"    ASR goc   : {src}",
            f"    Dap an    : {ref}",
            f"    Mo hinh   : {pred}",
        ]

    lines += [
        "\n" + "-" * 78,
        f"Tong ket: {counts['DUNG']} cau dung hoan toan | "
        f"{counts['GAN DUNG']} cau gan dung | {counts['SAI']} cau khong sua duoc",
        "-" * 78,
    ]
    return "\n".join(lines)


def build_log(
    args,
    results: Dict[str, Dict[str, float]],
    sources: List[str],
    references: List[str],
    predictions: Dict[str, List[str]],
    error_types: List[List[str]],
    engine: str,
) -> str:
    """Gom toàn bộ đánh giá thành một tệp .txt đọc được ngay, không cần mở Excel."""
    bar = "=" * 78
    parts = [
        bar,
        "LOG ĐÁNH GIÁ BENCHMARK ĐỀ TÀI NLP-28",
        "Sửa lỗi & chuẩn hóa phụ đề tiếng Việt từ ASR bằng mô hình Seq2Seq",
        bar,
        f"Thời điểm chạy : {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Thiết bị       : {'GPU/Colab' if engine == 'seq2seq' else engine}",
        f"Tập kiểm thử   : {args.test_file} ({len(sources)} mẫu)",
        f"Mô hình Seq2Seq: {args.model_name_or_path}",
        f"Engine đo được : {engine}",
        f"Batch size     : {args.batch_size}",
        "",
        bar,
        "1. BẢNG TỔNG HỢP CHỈ SỐ ĐỐI ĐẦU",
        bar,
        format_markdown_table(results, len(sources)),
    ]

    et_table = format_error_type_table(predictions, sources, references, error_types)
    if et_table:
        parts += ["", bar, "2. HIỆU NĂNG TÁCH THEO LOẠI LỖI", bar, et_table]

    parts += ["", bar, "3. CHI TIẾT SỐ CÂU THUỘC TỪNG NHÓM SỬA LỖI", bar]
    for name, m in results.items():
        parts.append(
            f"  {name:<34} sua dung {m['n_sua_dung']:>3} | "
            f"sua nham {m['n_sua_nham']:>3} | bo sot {m['n_bo_sot']:>3} | "
            f"giu dung {m['n_giu_dung']:>3}"
        )
    parts.append("")
    parts.append("  ('sua nham' gom ca over-correction: câu vốn đúng bị sửa tới.)")

    parts += ["", format_case_review(predictions, sources, references, error_types)]
    parts += ["", bar, "HẾT LOG", bar, ""]
    return "\n".join(parts)


def export_to_excel(
    results: Dict[str, Dict[str, float]],
    sources: List[str],
    references: List[str],
    predictions: Dict[str, List[str]],
    error_types: List[List[str]],
    output_excel_path: str,
):
    """Xuất báo cáo benchmark ra tệp Excel (.xlsx) gồm 2 sheet trực quan:
    1. 'TongHop_ChiSo': bảng tổng hợp mọi chỉ số của cả 3 phương pháp.
    2. 'ChiTiet_TungCau': từng câu — gốc ASR, đáp án chuẩn, kết quả mỗi mô hình.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[Cảnh báo] Chưa cài đặt pandas, không thể xuất file Excel.")
        return

    out_path = Path(output_excel_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name, metrics in results.items():
        row = {"Phương pháp đối chứng": model_name}
        for key, label in METRIC_COLUMNS:
            row[label] = metrics[key]
        row["Số câu sửa đúng"] = metrics["n_sua_dung"]
        row["Số câu sửa nhầm"] = metrics["n_sua_nham"]
        row["Số câu bỏ sót"] = metrics["n_bo_sot"]
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)

    detail_rows = []
    for idx, (src, ref) in enumerate(zip(sources, references)):
        row = {
            "STT": idx + 1,
            "Loại lỗi": ", ".join(error_types[idx]),
            "Câu gốc ASR (Noisy)": src,
            "Đáp án chuẩn (Ground Truth)": ref,
        }
        for model_name, preds in predictions.items():
            row[f"Kết quả {model_name}"] = preds[idx] if idx < len(preds) else ""
        detail_rows.append(row)
    df_detail = pd.DataFrame(detail_rows)

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

    sources, references, error_types = load_test_data(args.test_file)
    print(f"- Số lượng mẫu test: {len(sources)} câu\n")

    all_results: Dict[str, Dict[str, float]] = {}
    all_predictions: Dict[str, List[str]] = {}

    # 1. Baseline 1: Identity
    print("[1/3] Đang đánh giá Baseline 1: Identity (giữ nguyên)...")
    preds_b1 = IdentityBaseline().predict_batch(sources)
    res_b1 = evaluate_predictions(sources, preds_b1, references)
    all_results["Baseline 1 (Identity)"] = res_b1
    all_predictions["Baseline 1 (Identity)"] = preds_b1
    print(f"      -> BLEU: {res_b1['bleu']} | WER: {res_b1['wer']}% | CER: {res_b1['cer']}% | Edit-F1: {res_b1['edit_f1']}%")

    # 2. Baseline 2: Rule & Dictionary
    print("[2/3] Đang đánh giá Baseline 2: Rule & Dictionary...")
    preds_b2 = RuleAndDictionaryBaseline().predict_batch(sources)
    res_b2 = evaluate_predictions(sources, preds_b2, references)
    all_results["Baseline 2 (Rule & Dictionary)"] = res_b2
    all_predictions["Baseline 2 (Rule & Dictionary)"] = preds_b2
    print(f"      -> BLEU: {res_b2['bleu']} | WER: {res_b2['wer']}% | CER: {res_b2['cer']}% | Edit-F1: {res_b2['edit_f1']}%")

    # 3. Mô hình Seq2Seq
    seq2seq_name = f"Seq2Seq ({Path(args.model_name_or_path).name})"
    print(f"[3/3] Đang đánh giá Seq2Seq ({args.model_name_or_path})...")
    from ..models.inference import Seq2SeqCorrector

    seq2seq = Seq2SeqCorrector(
        model_name_or_path=args.model_name_or_path,
        use_fallback=args.allow_fallback,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    engine = "seq2seq" if seq2seq.model is not None else "rule-baseline"
    if engine != "seq2seq":
        print("      !!! Cảnh báo: đang đo RULE BASELINE chứ không phải Seq2Seq.")
    else:
        print(f"      -> Đã nạp mô hình thật, thiết bị: {seq2seq.device}")

    preds_seq2seq = seq2seq.predict_batch(sources, batch_size=args.batch_size)
    res_seq2seq = evaluate_predictions(sources, preds_seq2seq, references)
    all_results[seq2seq_name] = res_seq2seq
    all_predictions[seq2seq_name] = preds_seq2seq
    print(f"      -> BLEU: {res_seq2seq['bleu']} | WER: {res_seq2seq['wer']}% | CER: {res_seq2seq['cer']}%"
          f" | Exact: {res_seq2seq['exact_match']}% | Edit-F1: {res_seq2seq['edit_f1']}%")

    # Bảng Markdown
    table_md = format_markdown_table(all_results, len(sources))
    et_md = format_error_type_table(all_predictions, sources, references, error_types)
    print("\n" + "=" * 65)
    print(table_md)
    if et_md:
        print(et_md)
    print("=" * 65)

    # 1. Báo cáo Markdown
    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table_md + "\n" + ("\n" + et_md + "\n" if et_md else ""))
        print(f"\n[Thông báo] Đã lưu bảng kết quả Markdown tại: {out_path}")

    # 2. Log text đầy đủ
    if args.output_log:
        log_path = Path(args.output_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_text = build_log(args, all_results, sources, references, all_predictions, error_types, engine)
        log_path.write_text(log_text, encoding="utf-8")
        print(f"[Thông báo] Đã lưu LOG ĐẦY ĐỦ tại: {log_path} ({len(log_text):,} ký tự)")

    # 3. Excel nhiều sheet
    if args.output_excel:
        export_to_excel(
            results=all_results,
            sources=sources,
            references=references,
            predictions=all_predictions,
            error_types=error_types,
            output_excel_path=args.output_excel,
        )

    # 4. JSON thô để so sánh máy đọc được giữa nhiều đợt train
    if args.output_json:
        payload = {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "test_file": args.test_file,
            "num_samples": len(sources),
            "model_name_or_path": args.model_name_or_path,
            "engine": engine,
            "device": getattr(seq2seq, "device", "unknown"),
            "decoding": dict(seq2seq.generate_kwargs) if engine == "seq2seq" else None,
            "batch_size": args.batch_size,
            "results": all_results,
        }
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Thông báo] Đã lưu metrics JSON tại: {json_path}")

    return all_results


if __name__ == "__main__":
    run_benchmark()
