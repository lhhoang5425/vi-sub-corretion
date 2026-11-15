"""So sánh cạnh nhau nhiều đợt train (A/B test) cho đề tài NLP-28.

Mỗi đợt train để lại trong thư mục của nó:
- manifest.json  : tham số huấn luyện, số câu, eval loss từng epoch, thời lượng
- metrics.json   : BLEU / WER / CER / Exact Match / ROUGE-L / Edit P-R-F1 + cấu hình giải mã

Script này gom hai tệp đó của mọi đợt rồi đặt lên cùng một bảng, trả lời câu hỏi
"cấu hình nào thắng, và thắng ở chỉ số nào" thay vì phải mở tay từng báo cáo.

Dùng:
    python -m src.evaluation.compare_runs --runs_root checkpoints
    python -m src.evaluation.compare_runs --runs_root checkpoints --output_md checkpoints/so_sanh.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# (khoá trong metrics.json, nhãn cột, hướng tốt hơn)
METRIC_COLUMNS = [
    ("exact_match", "EM%", "up"),
    ("bleu", "BLEU", "up"),
    ("wer", "WER%", "down"),
    ("cer", "CER%", "down"),
    ("rouge_l", "R-L%", "up"),
    ("edit_precision", "Ed-P%", "up"),
    ("edit_recall", "Ed-R%", "up"),
    ("edit_f1", "Ed-F1%", "up"),
]

CONFIG_COLUMNS = [
    ("train_size", "Câu", "{}"),
    ("epochs", "Ep", "{}"),
    ("batch", "Bs", "{}"),
    ("lr", "LR", "{}"),
    ("smooth", "LS", "{}"),
    ("pad", "Pad", "{}"),
    ("best_eval_loss", "EvalLoss", "{:.5f}"),
    ("minutes", "Phút", "{:.1f}"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="So sánh song song nhiều đợt train NLP-28")
    parser.add_argument("--runs_root", type=str, required=True,
                        help="Thư mục chứa các thư mục đợt train (mỗi cái có manifest.json)")
    parser.add_argument("--output_md", type=str, default=None,
                        help="Ghi bảng so sánh ra tệp Markdown (mặc định chỉ in ra màn hình)")
    parser.add_argument("--sort_by", type=str, default="exact_match",
                        help="Chỉ số dùng để xếp hạng đợt train")
    return parser.parse_args()


def collect_runs(runs_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue

        info = json.loads(manifest_path.read_text(encoding="utf-8"))
        hp = info.get("hyperparams", {})
        rs = info.get("results", {})

        row: Dict = {
            "run": d.name,
            "path": d,
            "base_model": str(info.get("base_model", "?")).split("/")[-1],
            "train_size": info.get("dataset", {}).get("train_size"),
            "epochs": hp.get("num_train_epochs"),
            "batch": hp.get("per_device_train_batch_size"),
            "lr": hp.get("learning_rate"),
            "smooth": hp.get("label_smoothing", 0.0),
            "pad": hp.get("padding", "static"),
            "best_eval_loss": rs.get("best_eval_loss"),
            "eval_curve": rs.get("eval_loss_per_epoch"),
            "minutes": round((rs.get("train_runtime_sec") or 0) / 60, 1),
            "metrics": {},
            "engine": None,
            "decoding": None,
            "n_test": None,
        }

        metrics_path = d / "metrics.json"
        if metrics_path.exists():
            bench = json.loads(metrics_path.read_text(encoding="utf-8"))
            row["engine"] = bench.get("engine")
            row["decoding"] = bench.get("decoding")
            row["n_test"] = bench.get("num_samples")
            row["device"] = bench.get("device")
            for name, m in (bench.get("results") or {}).items():
                if name.startswith("Seq2Seq"):
                    row["metrics"] = m
                    row["model_label"] = name
        rows.append(row)
    return rows


def best_values(rows: List[Dict]) -> Dict[str, float]:
    """Giá trị tốt nhất của từng chỉ số dọc theo các đợt, theo đúng chiều ↑/↓."""
    best: Dict[str, float] = {}
    for key, _, direction in METRIC_COLUMNS:
        vals = [r["metrics"].get(key) for r in rows if r["metrics"].get(key) is not None]
        if not vals:
            continue
        best[key] = min(vals) if direction == "down" else max(vals)
    return best


def sort_rows(rows: List[Dict], sort_by: str) -> List[Dict]:
    def key(r):
        m = r["metrics"]
        if not m:
            return -1e9
        val = m.get(sort_by)
        if val is None:
            return -1e9
        # WER/CER càng thấp càng tốt -> đảo dấu
        return -val if sort_by in ("wer", "cer") else val
    return sorted(rows, key=key, reverse=True)


def render_text(rows: List[Dict], best: Dict[str, float]) -> str:
    """Bảng cố định chiều rộng, đọc thẳng trên console Colab."""
    lines = []
    lines.append("=" * 118)
    lines.append("BẢNG SO SÁNH CÁC ĐỢT TRAIN (mỗi dòng một cấu hình)")
    lines.append("=" * 118)

    header = f"{'ĐỢT TRAIN':<30}{'Câu':>8}{'Ep':>4}{'Bs':>4}{'LS':>6}{'Pad':>9}{'EvalLoss':>10}{'Phút':>7}"
    header += "".join(f"{label:>8}" for _, label, _ in METRIC_COLUMNS)
    lines.append(header)
    lines.append("-" * 118)

    for r in rows:
        loss_txt = f"{r['best_eval_loss']:.5f}" if r["best_eval_loss"] is not None else "?"
        line = f"{r['run'][:29]:<30}"
        line += f"{(r['train_size'] or 0):>8,}"
        line += f"{str(r['epochs'] or '?'):>4}"
        line += f"{str(r['batch'] or '?'):>4}"
        line += f"{str(r['smooth']):>6}"
        line += f"{str(r['pad'])[:8]:>9}"
        line += f"{loss_txt:>10}"
        line += f"{r['minutes']:>7.1f}"

        m = r["metrics"]
        for key, _, _ in METRIC_COLUMNS:
            val = m.get(key)
            if val is None:
                line += f"{'-':>8}"
            elif best.get(key) is not None and abs(val - best[key]) < 1e-9:
                line += f"{(chr(42) + format(val, '.2f')):>8}"
            else:
                line += f"{format(val, '.2f'):>8}"
        lines.append(line)

    lines.append("-" * 118)
    lines.append("(*) = giá trị TỐT NHẤT cột đó. Chiều: EM/BLEU/R-L/Ed-P/Ed-R/Ed-F1 càng CAO càng tốt; WER/CER càng THẤP càng tốt.")
    lines.append("")
    lines.append("!! Cột EvalLoss KHÔNG so sánh được qua label_smoothing: LS > 0 trộn thêm thành phần")
    lines.append("   làm mềm nhãn vào loss nên số LUÔN cao hơn, dù mô hình tốt hơn. Muốn biết mô hình")
    lines.append("   nào giỏi hơn, đọc EM%/BLEU/WER/CER/Edit-F1 - những chỉ số đo trên câu thật.")
    return "\n".join(lines)


def render_markdown(rows: List[Dict], best: Dict[str, float]) -> str:
    md = ["## Bảng so sánh các đợt train (NLP-28)\n"]
    md.append("| Đợt train | Câu train | Eval loss | Phút | " +
              " | ".join(label for _, label, _ in METRIC_COLUMNS) + " |")
    md.append("| :--- | ---: | ---: | ---: | " + " | ".join(":---:" for _ in METRIC_COLUMNS) + " |")

    for r in rows:
        m = r["metrics"]
        cells = []
        for key, _, _ in METRIC_COLUMNS:
            val = m.get(key)
            if val is None:
                cells.append("-")
            elif best.get(key) is not None and abs(val - best[key]) < 1e-9:
                cells.append(f"**{val:.2f}**")
            else:
                cells.append(f"{val:.2f}")
        loss = f"{r['best_eval_loss']:.5f}" if r["best_eval_loss"] is not None else "?"
        md.append(f"| {r['run']} | {r['train_size'] or 0:,} | {loss} | {r['minutes']:.1f} | " +
                  " | ".join(cells) + " |")

    md.append("")
    md.append("**Chú thích hướng chỉ số:** EM% (Exact Match) và ROUGE-L càng cao càng tốt; "
              "WER%/CER% càng thấp càng tốt; Edit-P là tỷ lệ sửa đúng đích, Edit-R là tỷ lệ bao phủ câu cần sửa.")
    md.append("")
    md.append("**Cảnh báo đọc cột EvalLoss:** không so sánh được qua `label_smoothing`. LS > 0 trộn thành phần "
              "làm mềm nhãn vào hàm mất mát nên giá trị này LUÔN cao hơn, kể cả khi mô hình thực sự giỏi hơn. "
              "Kết luận hơn-thua phải dựa trên EM%/BLEU/WER/CER/Edit-F1.")
    md.append("")
    md.append("### Ghi chú cấu hình từng đợt")
    md.append("")
    for r in rows:
        dec = r["decoding"] or {}
        md.append(f"- **{r['run']}** — nền `{r['base_model']}`, pad {r['pad']}, label_smoothing {r['smooth']}, "
                  f"LR {r['lr']}, epoch curve {r['eval_curve']}, "
                  f"engine đo = **{r['engine'] or 'CHƯA BENCHMARK'}**, "
                  f"giải mã beams={dec.get('num_beams')} rep_pen={dec.get('repetition_penalty')} "
                  f"len_pen={dec.get('length_penalty')}, test {r['n_test'] or '-'} mẫu")
    return "\n".join(md) + "\n"


def warn(rows: List[Dict]) -> Optional[str]:
    """Báo động nếu một đợt không thực sự đo bằng Seq2Seq, hoặc test set lệch nhau."""
    problems = []
    for r in rows:
        if r["engine"] is None:
            problems.append(f"  - {r['run']}: chưa có metrics.json, chưa được benchmark.")
        elif r["engine"] != "seq2seq":
            problems.append(f"  - {r['run']}: engine = '{r['engine']}' -> KHÔNG phải mô hình Seq2Seq, số liệu đừng đem so.")
    sample_sizes = {r["n_test"] for r in rows if r["n_test"]}
    if len(sample_sizes) > 1:
        problems.append(f"  - Các đợt đo trên tập test khác kích thước {sorted(sample_sizes)} -> bảng thiếu công bằng.")
    devices = {r.get("device") for r in rows if r.get("device")}
    if len(devices) > 1:
        problems.append(f"  - Đo trộn lẫn thiết bị {sorted(devices)} (GPU fp16 và CPU fp32 cho số lệch nhau).")
    if not problems:
        return None
    return "\n!!! CẢNH BÁO TRƯỚC KHI ĐỌC BẢNG:\n" + "\n".join(problems)


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {runs_root}")

    rows = sort_rows(collect_runs(runs_root), args.sort_by)
    if not rows:
        print(f"Chưa có đợt train nào có manifest.json trong {runs_root}")
        print("Chạy train trước (mỗi đợt tự ghi manifest.json + metrics.json vào thư mục của nó).")
        return

    best = best_values(rows)
    text_table = render_text(rows, best)
    warning = warn(rows)
    if warning:
        print(warning)
    print(text_table)

    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            (warning.replace("! ", "- ") + "\n\n" if warning else "") +
            render_markdown(rows, best),
            encoding="utf-8",
        )
        print(f"\n[Thông báo] Đã ghi bảng so sánh Markdown tại: {md_path}")


if __name__ == "__main__":
    main()
