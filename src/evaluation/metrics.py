"""Mô-đun đánh giá hiệu năng (Evaluation Metrics)

Tính toán các chỉ số bắt buộc theo yêu cầu đề cương NLP-28:
1. WER (Word Error Rate): Tỷ lệ sai lệch từ so với câu chuẩn.
2. BLEU / SacreBLEU: Độ tương đồng câu sinh so với câu tham chiếu.
3. Edit-level Precision, Recall, F1: Đánh giá độ chính xác của các chỉnh sửa (tránh sửa sai câu đúng).
"""

from typing import Dict, List
import sacrebleu


def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """Tính điểm SacreBLEU trên danh sách câu dự đoán và câu tham chiếu."""
    if not predictions or not references:
        return 0.0
    bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
    return round(bleu.score, 2)


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """Tính tỷ lệ lỗi từ vựng (Word Error Rate)."""
    try:
        from jiwer import wer
        return round(float(wer(references, predictions)) * 100, 2)
    except ImportError:
        # Fallback tính WER đơn giản nếu chưa cài jiwer
        total_words = 0
        total_mismatch = 0
        for p, r in zip(predictions, references):
            p_words = p.split()
            r_words = r.split()
            total_words += max(len(r_words), 1)
            # Đếm số từ khác nhau đơn giản
            diff = abs(len(p_words) - len(r_words)) + sum(1 for w1, w2 in zip(p_words, r_words) if w1 != w2)
            total_mismatch += diff
        return round((total_mismatch / max(total_words, 1)) * 100, 2)


def calculate_edit_metrics(
    sources: List[str], predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Tính Edit-level Precision, Recall, F1.

    Đo lường mức độ sửa đúng tại các vị trí cần sửa, phát hiện lỗi over-correction.
    """
    tp, fp, fn = 0, 0, 0

    for src, pred, ref in zip(sources, predictions, references):
        need_edit = src.strip() != ref.strip()
        made_edit = src.strip() != pred.strip()

        if need_edit and made_edit:
            if pred.strip() == ref.strip():
                tp += 1  # Sửa đúng hoàn toàn
            else:
                fp += 1  # Có sửa nhưng chưa đúng đích
        elif not need_edit and made_edit:
            fp += 1  # Câu vốn đã đúng nhưng lại sửa nhầm (over-correction)
        elif need_edit and not made_edit:
            fn += 1  # Cần sửa nhưng lại giữ nguyên

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "edit_precision": round(precision * 100, 2),
        "edit_recall": round(recall * 100, 2),
        "edit_f1": round(f1 * 100, 2),
    }


def evaluate_predictions(
    sources: List[str], predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Tổng hợp toàn bộ chỉ số đánh giá thành một từ điển kết quả."""
    results = {}
    results["bleu"] = calculate_bleu(predictions, references)
    results["wer"] = calculate_wer(predictions, references)
    edit_metrics = calculate_edit_metrics(sources, predictions, references)
    results.update(edit_metrics)
    return results


if __name__ == "__main__":
    sources = ["toi di lam", "hom nay rat dep"]
    predictions = ["Tôi đi làm.", "Hôm nay rất đẹp."]
    references = ["Tôi đi làm.", "Hôm nay rất đẹp!"]

    eval_res = evaluate_predictions(sources, predictions, references)
    print("Kết quả đánh giá mẫu:", eval_res)
