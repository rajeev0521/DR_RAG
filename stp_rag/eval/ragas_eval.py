"""RAGAS metric wrapper for context and answer quality assessment."""

from __future__ import annotations

from typing import Dict, List, Optional


def evaluate_ragas(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Computes RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall.
    Returns zero dict if ragas / LLM judge is not configured or in lightweight testing.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        res = evaluate(dataset, metrics=metrics)
        return {
            "ragas_faithfulness": float(res.get("faithfulness", 0.0)),
            "ragas_answer_relevancy": float(res.get("answer_relevancy", 0.0)),
            "ragas_context_precision": float(res.get("context_precision", 0.0)),
            "ragas_context_recall": float(res.get("context_recall", 0.0)),
        }
    except Exception:
        # Fallback for environments without OpenAI / LLM judge API key configured
        return {
            "ragas_faithfulness": 0.0,
            "ragas_answer_relevancy": 0.0,
            "ragas_context_precision": 0.0,
            "ragas_context_recall": 0.0,
        }
