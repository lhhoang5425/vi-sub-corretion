"""Script huấn luyện mô hình Seq2Seq (ViT5 hoặc BARTpho)

Sử dụng Hugging Face Seq2SeqTrainer và DataCollatorForSeq2Seq
Có thể chạy trực tiếp trên máy có GPU hoặc Google Colab.

Mỗi đợt train được lưu vào thư mục RIÊNG kèm manifest.json ghi lại toàn bộ tham
số và chỉ số. Đợt sau không bao giờ đè mất đợt trước, và có thể resume khi Colab
thu hồi runtime giữa chừng.
"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ..data_processing.dataset_loader import load_correction_dataset, prepare_seq2seq_features


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Seq2Seq sửa lỗi phụ đề tiếng Việt")
    parser.add_argument("--model_name", type=str, default="bmd1905/vietnamese-correction-v2", help="Model nền (bmd1905/vietnamese-correction-v2 hoặc vinai/bartpho-syllable)")
    parser.add_argument("--train_file", type=str, default="data/synthetic/train.jsonl", help="Đường dẫn file train")
    parser.add_argument("--val_file", type=str, default="data/synthetic/val.jsonl", help="Đường dẫn file validation")
    parser.add_argument("--output_dir", type=str, default="checkpoints/bmd1905-sub-corrector", help="Thư mục lưu weights; khi chạy Colab nên trỏ thẳng vào Google Drive")
    parser.add_argument("--run_tag", type=str, default=None, help="Nhãn đợt chạy, ví dụ '10k' / '100k'. Được ghép thành hậu tố của tên thư mục.")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=3e-5, help="Tốc độ học")
    parser.add_argument("--max_length", type=int, default=128, help="Trần độ dài chuỗi (token); câu ngắn hơn được pad động theo batch")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing cho cross-entropy. 0.1 thường giảm over-correction vì mô hình bớt confident vào một từ duy nhất")
    parser.add_argument("--static_padding", action="store_true", help="Quay lại pad cứng về max_length (hành vi cũ). Mặc định pad ĐỘNG theo batch")
    parser.add_argument("--num_train_split", type=str, default=None, help="Chỉ dùng một phần tập train, vd '20%%' để test nhanh cấu hình trước khi chạy full")
    parser.add_argument("--overwrite", action="store_true", help="Cố ý ghi đè đợt train cũ. MẶC ĐỊNH TẮT để bảo vệ checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục đợt train dở dang từ epoch checkpoint mới nhất trong thư mục đích")
    parser.add_argument("--prune_checkpoints", action="store_true", help="XÓA checkpoint epoch trung gian sau khi train xong để tiết kiệm chỗ. MẶC ĐỊNH GIỮ LẠI toàn bộ vì mỗi đợt là bằng chứng thực nghiệm.")
    parser.add_argument("--list_runs", action="store_true", help="Chỉ liệt kê các đợt train đã có trong thư mục chứa output_dir rồi thoát")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1024 / 1024, 1)


def resolve_output_dir(output_dir: str, run_tag: Optional[str], overwrite: bool) -> Path:
    """Chọn thư mục đích chưa từng dùng, để đợt train sau không đè mất đợt trước."""
    base = Path(output_dir)
    if run_tag:
        base = base.with_name(f"{base.name}_{run_tag}")

    if not base.exists():
        return base

    if overwrite:
        print(f"[CẢNH BÁO] --overwrite đang bật: nội dung cũ của {base} sẽ bị thay thế.")
        return base

    version = 2
    while base.with_name(f"{base.name}_v{version}").exists():
        version += 1
    candidate = base.with_name(f"{base.name}_v{version}")
    print(f"[AN TOÀN] {base} đã tồn tại từ một đợt train trước.")
    print(f"[AN TOÀN] Tự động chuyển sang {candidate} để KHÔNG mất checkpoint cũ.")
    print("           Muốn đè hẳn: thêm --overwrite. Muốn đặt tên riêng: thêm --run_tag.")
    return candidate


def latest_epoch_checkpoint(out_dir: Path) -> Optional[Path]:
    ckpts = [p for p in out_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.name.split("-", 1)[1]))


def list_runs(output_dir: str) -> None:
    """In bảng đối chiếu mọi đợt train đã có trong thư mục cha của output_dir."""
    root = Path(output_dir).parent
    header = ("ĐỢT TRAIN", "MÔ HÌNH NỀN", "CÂU", "EPOCH", "LR", "EVAL_LOSS", "MB")
    rows: List[tuple] = []

    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        size_mb = _dir_size_mb(d)
        manifest = d / "manifest.json"
        if manifest.exists():
            info = json.loads(manifest.read_text(encoding="utf-8"))
            hp, ds, rs = info.get("hyperparams", {}), info.get("dataset", {}), info.get("results", {})
            rows.append((
                d.name,
                str(info.get("base_model", "?")).split("/")[-1],
                f"{ds.get('train_size', 0):,}",
                hp.get("num_train_epochs", "?"),
                hp.get("learning_rate", "?"),
                rs.get("best_eval_loss", "?"),
                size_mb,
            ))
        elif (d / "config.json").exists():
            rows.append((d.name, "?", "?", "?", "?", "CHƯA XONG", size_mb))

    if not rows:
        print(f"Chưa có đợt train nào trong {root}")
        return

    line = "-" * 104
    print(f"\n{header[0]:<34}{header[1]:<22}{header[2]:>10}{header[3]:>6}{header[4]:>10}{header[5]:>11}{header[6]:>9}")
    print(line)
    for row in rows:
        print(f"{row[0]:<34}{row[1]:<22}{row[2]:>10}{str(row[3]):>6}{str(row[4]):>10}{str(row[5]):>11}{row[6]:>9}")
    print(line)
    total_mb = round(sum(r[-1] for r in rows), 1)
    print(f"Tổng: {len(rows)} đợt, {total_mb:,} MB trong {root}")
    print(f"Google Drive miễn phí hạn mức 15,360 MB -> còn trống {max(15360 - total_mb, 0):,.0f} MB")


def train():
    args = parse_args()

    if args.list_runs:
        list_runs(args.output_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"=== BẮT ĐẦU HUẤN LUYỆN: {args.model_name} ===")
    print(f"Thiết bị: {device} ({gpu_name})")

    out_dir = resolve_output_dir(args.output_dir, args.run_tag, args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[AN TOÀN] Đợt này lưu tại: {out_dir}")

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    if not train_path.exists():
        print(f"[LỖI] Chưa có dữ liệu train tại {args.train_file}!")
        print("Gợi ý: Hãy chạy script sinh dữ liệu noise_generator.py trước.")
        return

    # 1. Nạp tokenizer và model (có fallback use_fast=False khi gặp lỗi Rust Unigram)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"[Thông báo] Fast tokenizer không tương thích ({e}), chuyển sang use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 2. Đọc dataset
    raw_datasets = load_correction_dataset(
        train_path=train_path,
        val_path=val_path if val_path.exists() else None
    )
    if args.num_train_split and "train" in raw_datasets:
        frac = float(str(args.num_train_split).strip("%")) / 100.0
        raw_datasets["train"] = raw_datasets["train"].select(
            range(max(int(len(raw_datasets["train"]) * frac), 1))
        )

    train_size = len(raw_datasets["train"])
    val_size = len(raw_datasets["validation"]) if "validation" in raw_datasets else 0
    print(f"[DỮ LIỆU] Train: {train_size:,} câu | Validation: {val_size:,} câu")

    # 3. Tiền xử lý tokenization
    pad_mode = "static" if args.static_padding else "dynamic"
    tokenized_datasets = raw_datasets.map(
        lambda ex: prepare_seq2seq_features(
            ex,
            tokenizer,
            args.max_length,
            args.max_length,
            dynamic_padding=(pad_mode == "dynamic"),
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    print(f"[TOKENIZATION] Pad {'ĐỘNG theo batch (mặc định mới)' if pad_mode == 'dynamic' else 'CỨNG về ' + str(args.max_length) + ' token (hành vi cũ)'}"
          f" | trần {args.max_length} token")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    resume_from = None
    if args.resume:
        resume_from = latest_epoch_checkpoint(out_dir)
        if resume_from is not None:
            print(f"[RESUME] Tiếp tục từ epoch checkpoint: {resume_from.name}")
        else:
            print(f"[RESUME] {out_dir} không có checkpoint-* nào, bắt đầu lại từ đầu.")
            resume_from = None

    # 4. Cấu hình huấn luyện (Hỗ trợ cả transformers mới 'eval_strategy' và cũ 'evaluation_strategy')
    training_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "predict_with_generate": True,
        "save_strategy": "epoch",
        # None = giữ mọi checkpoint epoch, không để thư viện tự xóa.
        # Mỗi checkpoint ~4.7 GB (weights + optimizer state), Drive 5 TB chứa dư
        # vài chục đợt, và chúng là thứ cho phép --resume khi Colab rớt runtime.
        "save_total_limit": None,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "label_smoothing": args.label_smoothing,
        "fp16": device == "cuda",
        "logging_steps": 50,
        "report_to": "none",
    }
    eval_val = "epoch" if val_size else "no"
    try:
        training_args = Seq2SeqTrainingArguments(**training_kwargs, eval_strategy=eval_val)
    except TypeError:
        training_args = Seq2SeqTrainingArguments(**training_kwargs, evaluation_strategy=eval_val)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets.get("validation", None),
        "data_collator": data_collator,
    }
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=tokenizer)

    # 5. Huấn luyện và lưu model
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 6. Ghi manifest để đối chiếu giữa các đợt train mà không phải đọc lại log
    log = trainer.state.log_history
    eval_losses = [round(h["eval_loss"], 5) for h in log if "eval_loss" in h]
    grad_norms = [h["grad_norm"] for h in log if "grad_norm" in h]
    runtime = train_result.metrics.get("train_runtime")

    manifest = {
        "run_name": out_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "base_model": args.model_name,
        "device": device,
        "gpu": gpu_name,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "hyperparams": {
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "weight_decay": 0.01,
            "label_smoothing": args.label_smoothing,
            "padding": pad_mode,
            "fp16": device == "cuda",
        },
        "dataset": {
            "train_file": str(train_path),
            "val_file": str(val_path) if val_size else None,
            "train_size": train_size,
            "val_size": val_size,
        },
        "results": {
            "train_loss_final": round(train_result.metrics.get("train_loss", 0.0), 5),
            "eval_loss_per_epoch": eval_losses,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "max_grad_norm": round(max(grad_norms), 4) if grad_norms else None,
            "train_runtime_sec": round(runtime, 1) if runtime else None,
            "train_samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0.0), 2),
            "total_steps": int(trainer.state.global_step),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 7. Ghi log train chi tiết từng step xuống cùng thư mục đợt train
    (out_dir / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    readable = ["=" * 66, "LOG HUẤN LUYỆN " + out_dir.name, "=" * 66]
    for key, value in manifest["results"].items():
        readable.append(f"{key:<28} {value}")
    readable.append("")
    readable.append(f"{'STEP':>8} {'EPOCH':>8} {'LOSS':>10} {'GRAD_NORM':>11} {'LR':>12}")
    readable.append("-" * 66)
    for h in trainer.state.log_history:
        if "loss" not in h:
            continue
        readable.append(
            f"{h.get('step', 0):>8} {h.get('epoch', 0):>8.3f} {h['loss']:>10.5f} "
            f"{h.get('grad_norm', float('nan')):>11.4f} {h.get('learning_rate', 0.0):>12.3e}"
        )
    readable.append("")
    for h in trainer.state.log_history:
        if "eval_loss" in h:
            readable.append(f"  epoch {h.get('epoch', '?')}: eval_loss = {h['eval_loss']:.5f}")
    (out_dir / "train_log.txt").write_text("\n".join(readable) + "\n", encoding="utf-8")

    # 8. Checkpoint epoch: MẶC ĐỊNH GIỮ LẠI hết (Drive 5 TB dư sức, và chúng là bằng chứng
    #    thực nghiệm cho từng epoch + điểm --resume nếu Colab rớt runtime giữa đợt).
    kept = sorted(
        (p for p in out_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    if args.prune_checkpoints:
        freed_mb = sum(_dir_size_mb(c) for c in kept)
        for ckpt in kept:
            shutil.rmtree(ckpt)
        print(f"[DỌN DẸP] Đã xóa {len(kept)} checkpoint epoch, giải phóng {freed_mb:,} MB.")
        kept = []
    else:
        print(f"[GIỮ LẠI] {len(kept)} checkpoint epoch vẫn nằm trong đợt train này: "
              + ", ".join(c.name for c in kept))

    manifest["checkpoints_kept"] = [c.name for c in kept]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    size_mb = _dir_size_mb(out_dir)
    print("=== ĐÃ HUẤN LUYỆN XONG ===")
    print(f"Thư mục đợt train : {out_dir}")
    print(f"Dung lượng         : {size_mb:,} MB")
    print(f"Train loss cuối    : {manifest['results']['train_loss_final']}")
    print(f"Eval loss/epoch    : {eval_losses}")
    print(f"Steps / thời lượng : {manifest['results']['total_steps']} steps"
          + (f", {round(runtime / 60, 1)} phút" if runtime else ""))
    print(f"Manifest           : {out_dir / 'manifest.json'}")
    print(f"Log đọc được       : {out_dir / 'train_log.txt'}")
    print("Bước tiếp theo: chạy benchmark NGAY trên GPU này (đừng kéo về máy cá nhân,")
    print("                CPU local vừa chậm vừa khác fp16 nên số liệu sẽ lệch log Colab).")


if __name__ == "__main__":
    train()
