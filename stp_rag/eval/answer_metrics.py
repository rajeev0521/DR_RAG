"""Answer quality metrics: Exact Match (EM) and Token-level F1."""

from __future__ import annotations

import re
import string
from typing import Sequence


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def compute_qa_metrics(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> tuple[float, float]:
    """Computes mean EM and mean F1 over predictions."""
    if not predictions:
        return 0.0, 0.0

    em_total = 0.0
    f1_total = 0.0
    n = len(predictions)

    for p, g in zip(predictions, ground_truths):
        em_total += exact_match_score(p, g)
        f1_total += f1_score(p, g)

    return em_total / n, f1_total / n
