"""Mô-đun tải và chuẩn bị dữ liệu (Dataset Loader)

Hỗ trợ đọc từ JSONL/CSV và định dạng tokenization cho Seq2SeqTrainer
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    Dataset = Any
    DatasetDict = Any


def load_correction_dataset(
    train_path: Optional[Union[str, Path]] = None,
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
) -> DatasetDict:
    """Đọc dữ liệu cặp câu (noisy, clean) từ các tệp JSONL hoặc CSV."""
    dataset_dict = {}

    def _read_file(path: Union[str, Path]):
        if Dataset is Any:
            raise ImportError("Vui lòng cài đặt thư viện 'datasets' bằng lệnh: pip install datasets")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp: {path}")

        records = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            records = df.to_dict(orient="records")
        else:
            raise ValueError(f"Định dạng chưa được hỗ trợ: {path.suffix}")

        return Dataset.from_list(records)

    if train_path:
        dataset_dict["train"] = _read_file(train_path)
    if val_path:
        dataset_dict["validation"] = _read_file(val_path)
    if test_path:
        dataset_dict["test"] = _read_file(test_path)

    return DatasetDict(dataset_dict)


def prepare_seq2seq_features(examples, tokenizer, max_input_length=128, max_target_length=128):
    """Tiền xử lý tokenization cho Seq2Seq (ViT5 hoặc BARTpho)."""
    inputs = examples["noisy"]
    targets = examples["clean"]

    # Tokenize input
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )

    # Tokenize labels
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )

    # Thay thế id padding bằng -100 để không tính loss trên token padding
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
