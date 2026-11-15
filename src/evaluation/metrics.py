"""Mô-đun đánh giá hiệu năng (Evaluation Metrics)

Các chỉ số theo yêu cầu đề cương NLP-28:
1. WER (Word Error Rate): Tỷ lệ sai lệch ở mức từ.
2. CER (Character Error Rate): Tỷ lệ sai lệch ở mức ký tự. Tiếng Việt lỗi ở DẤU THANH
   và DẤU CÂU là lỗi ký tự, WER không thấy được (một từ sai dấu vẫn tính là 1 từ).
3. BLEU / SacreBLEU: Độ tương đồng câu sinh so với câu tham chiếu.
4. Exact Match: Tỷ lệ câu mô hình sinh ra trùng 100% đáp án chuẩn.
5. Edit-level Precision, Recall, F1: Độ chính xác tại các vị trí sửa lỗi,
   kèm đếm trực tiếp số câu sửa đúng / sửa nhầm / bỏ sót.
6. ROUGE-L: Dựa trên chuỗi con chung dài nhất, bù đắp điểm yếu của BLEU với câu ngắn.
"""

import collections
import math
from typing import Dict, List

try:
    import sacrebleu
except ImportError:
    sacrebleu = None


def _simple_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> float:
    """Fallback tính BLEU thuần Python đơn giản khi chưa cài sacrebleu."""
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        if not pred_tokens or not ref_tokens:
            continue

        # N-gram precisions
        p_n = []
        for n in range(1, max_n + 1):
            pred_ngrams = collections.Counter(
                tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
            )
            ref_ngrams = collections.Counter(
                tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
            )
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = max(len(pred_tokens) - n + 1, 1)
            p_n.append(max(overlap / total, 1e-9))

        # Brevity penalty
        bp = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
        geom_mean = math.exp(sum(math.log(p) for p in p_n) / max_n)
        total_score += bp * geom_mean * 100

    return round(total_score / max(len(predictions), 1), 2)


def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """Tính điểm SacreBLEU (hoặc fallback thuần Python nếu chưa cài sacrebleu)."""
    if not predictions or not references:
        return 0.0
    if sacrebleu is not None:
        bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
        return round(bleu.score, 2)
    return _simple_bleu(predictions, references)


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


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """Tính tỷ lệ lỗi ở mức ký tự (Character Error Rate).

    Chỉ số này quan trọng hơn WER với tiếng Việt: lỗi thiếu dấu thanh hay sai dấu câu
    đều nằm ở cấp ký tự, còn WER vẫn đếm cả từ đó là "1 từ sai" dù chỉ lệch một dấu.
    """
    if not predictions or not references:
        return 0.0
    try:
        from jiwer import cer
        return round(float(cer(references, predictions)) * 100, 2)
    except ImportError:
        # Fallback: khoảng cách Levenshtein trên từng cặp câu
        total_chars, total_dist = 0, 0
        for p, r in zip(predictions, references):
            p_ch, r_ch = p.strip(), r.strip()
            total_chars += max(len(r_ch), 1)
            prev = list(range(len(p_ch) + 1))
            for i, rc in enumerate(r_ch, start=1):
                cur = [i]
                for j, pc in enumerate(p_ch, start=1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != pc)))
                prev = cur
            total_dist += prev[-1]
        return round((total_dist / max(total_chars, 1)) * 100, 2)


def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """Tỷ lệ câu sinh ra trùng khớp tuyệt đối với đáp án chuẩn (%).

    Chỉ số dễ đọc nhất cho hội đồng: "mô hình phục hồi chính xác hoàn toàn X% số câu".
    """
    if not predictions:
        return 0.0
    hits = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return round(hits / len(references) * 100, 2)


def calculate_rouge_l(predictions: List[str], references: List[str]) -> float:
    """ROUGE-L F1 dựa trên chuỗi con chung dài nhất (LCS) ở mức từ."""
    if not predictions or not references:
        return 0.0

    def lcs_len(a: List[str], b: List[str]) -> int:
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b, start=1):
                cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
            prev = cur
        return prev[-1]

    scores = []
    for p, r in zip(predictions, references):
        pt, rt = p.strip().split(), r.strip().split()
        if not pt or not rt:
            scores.append(0.0)
            continue
        common = lcs_len(pt, rt)
        prec = common / len(pt)
        rec = common / len(rt)
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return round(sum(scores) / len(scores) * 100, 2)


def calculate_edit_metrics(
    sources: List[str], predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Tính Edit-level Precision, Recall, F1 kèm số câu thuộc từng nhóm.

    Phân loại theo hai trục: câu CÓ CẦN sửa hay không, và mô hình CÓ SỬA hay không.
    """
    tp, fp, fn, tn = 0, 0, 0, 0

    for src, pred, ref in zip(sources, predictions, references):
        need_edit = src.strip() != ref.strip()
        made_edit = src.strip() != pred.strip()

        if need_edit and made_edit:
            if pred.strip() == ref.strip():
                tp += 1  # Sửa đúng hoàn toàn
            else:
                fp += 1  # Có sửa nhưng chưa khớp đáp án chuẩn
        elif not need_edit and made_edit:
            fp += 1  # Over-correction: câu vốn đúng bị sửa nhầm
        elif need_edit and not made_edit:
            fn += 1  # Bỏ sót: cần sửa nhưng giữ nguyên
        else:
            tn += 1  # Câu đúng và được giữ nguyên (xử lý lịch sự)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "edit_precision": round(precision * 100, 2),
        "edit_recall": round(recall * 100, 2),
        "edit_f1": round(f1 * 100, 2),
        "n_sua_dung": tp,
        "n_sua_nham": fp,
        "n_bo_sot": fn,
        "n_giu_dung": tn,
    }


def evaluate_predictions(
    sources: List[str], predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Tổng hợp toàn bộ chỉ số đánh giá thành một từ điển kết quả."""
    results = {}
    results["bleu"] = calculate_bleu(predictions, references)
    results["wer"] = calculate_wer(predictions, references)
    results["cer"] = calculate_cer(predictions, references)
    results["exact_match"] = calculate_exact_match(predictions, references)
    results["rouge_l"] = calculate_rouge_l(predictions, references)
    results.update(calculate_edit_metrics(sources, predictions, references))
    return results


if __name__ == "__main__":
    sources = ["toi di lam", "hom nay rat dep"]
    predictions = ["Tôi đi làm.", "Hôm nay rất đẹp."]
    references = ["Tôi đi làm.", "Hôm nay rất đẹp!"]

    eval_res = evaluate_predictions(sources, predictions, references)
    for key, value in eval_res.items():
        print(f"  {key:<16} {value}")

