"""RAGAS metric wrapper for context and answer quality assessment."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def evaluate_ragas(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: List[str],
    judge_model: str = "phi3:mini",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    ollama_url: str = "http://localhost:11434",
) -> Dict[str, Optional[float]]:
    """
    Computes RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
    using local Ollama judge LLM and local HuggingFace embeddings.
    If judge evaluation fails or is unconfigured, returns None (explicit null) for all metrics.
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
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import HuggingFaceEmbeddings

        llm = ChatOllama(model=judge_model, base_url=ollama_url, temperature=0.0)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        res = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        logger.info(f"RAGAS evaluation succeeded using judge model '{judge_model}'")
        return {
            "ragas_faithfulness": float(res.get("faithfulness")) if res.get("faithfulness") is not None else None,
            "ragas_answer_relevancy": float(res.get("answer_relevancy")) if res.get("answer_relevancy") is not None else None,
            "ragas_context_precision": float(res.get("context_precision")) if res.get("context_precision") is not None else None,
            "ragas_context_recall": float(res.get("context_recall")) if res.get("context_recall") is not None else None,
        }
    except Exception as exc:
        logger.warning(f"RAGAS evaluation unavailable or failed: {exc}. Returning explicit null sentinels.")
        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_context_precision": None,
            "ragas_context_recall": None,
        }
