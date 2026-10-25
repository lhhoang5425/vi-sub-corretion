"""Script huấn luyện mô hình Seq2Seq (ViT5 hoặc BARTpho)

Sử dụng Hugging Face Seq2SeqTrainer và DataCollatorForSeq2Seq
Có thể chạy trực tiếp trên máy có GPU hoặc Google Colab.
"""

import argparse
from pathlib import Path
import torch
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
    parser.add_argument("--model_name", type=str, default="VietAI/vit5-base", help="Model nền (VietAI/vit5-base hoặc vinai/bartpho-syllable)")
    parser.add_argument("--train_file", type=str, default="data/synthetic/train.jsonl", help="Đường dẫn file train")
    parser.add_argument("--val_file", type=str, default="data/synthetic/val.jsonl", help="Đường dẫn file validation")
    parser.add_argument("--output_dir", type=str, default="checkpoints/vit5-sub-corrector", help="Thư mục lưu weights")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=5e-5, help="Tốc độ học")
    parser.add_argument("--max_length", type=int, default=128, help="Độ dài chuỗi tối đa")
    return parser.parse_args()


def train():
    args = parse_args()
    print(f"=== BẮT ĐẦU HUẤN LUYỆN: {args.model_name} ===")
    print(f"Thiết bị: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 1. Nạp tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 2. Đọc dataset
    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    if not train_path.exists():
        print(f"[LỖI] Chưa có dữ liệu train tại {args.train_file}!")
        print("Gợi ý: Hãy chạy script sinh dữ liệu noise_generator.py trước.")
        return

    raw_datasets = load_correction_dataset(
        train_path=train_path,
        val_path=val_path if val_path.exists() else None
    )

    # 3. Tiền xử lý tokenization
    tokenized_datasets = raw_datasets.map(
        lambda ex: prepare_seq2seq_features(ex, tokenizer, args.max_length, args.max_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4. Cấu hình huấn luyện
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch" if val_path.exists() else "no",
        save_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 5. Huấn luyện và lưu model
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"=== ĐÃ HUẤN LUYỆN XONG! Mô hình được lưu tại: {args.output_dir} ===")


if __name__ == "__main__":
    train()
