"""Retrieval evaluation metrics: Recall@k and Mean Reciprocal Rank (MRR)."""

from __future__ import annotations

from typing import List, Sequence, Set


def compute_recall_at_k(
    retrieved_chunk_ids_per_query: Sequence[Sequence[int | str]],
    gold_chunk_ids_per_query: Sequence[Set[int | str]],
    k: int,
) -> float:
    """Computes Recall@k across queries."""
    if not retrieved_chunk_ids_per_query:
        return 0.0

    hits = 0
    total = len(retrieved_chunk_ids_per_query)

    for retrieved, gold in zip(retrieved_chunk_ids_per_query, gold_chunk_ids_per_query):
        top_k = set(retrieved[:k])
        if any(gid in top_k for gid in gold):
            hits += 1

    return hits / total


def compute_mrr(
    retrieved_chunk_ids_per_query: Sequence[Sequence[int | str]],
    gold_chunk_ids_per_query: Sequence[Set[int | str]],
) -> float:
    """Computes Mean Reciprocal Rank (MRR)."""
    if not retrieved_chunk_ids_per_query:
        return 0.0

    rr_sum = 0.0
    total = len(retrieved_chunk_ids_per_query)

    for retrieved, gold in zip(retrieved_chunk_ids_per_query, gold_chunk_ids_per_query):
        reciprocal_rank = 0.0
        for rank_idx, cand_id in enumerate(retrieved, start=1):
            if cand_id in gold:
                reciprocal_rank = 1.0 / rank_idx
                break
        rr_sum += reciprocal_rank

    return rr_sum / total
