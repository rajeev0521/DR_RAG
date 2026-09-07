"""Efficiency metrics calculation."""

from __future__ import annotations

from typing import List, Sequence
from ..chunking.segment import Chunk


def compute_efficiency_metrics(
    chunks_per_document: Sequence[Sequence[Chunk]],
    total_build_time_sec: float,
    query_latencies_ms: Sequence[float],
) -> dict[str, float]:
    """Computes efficiency metrics for an experimental arm."""
    num_docs = len(chunks_per_document)
    if num_docs == 0:
        return {
            "chunks_per_doc": 0.0,
            "tokens_per_chunk": 0.0,
            "build_time_sec": total_build_time_sec,
            "retrieval_latency_ms": 0.0,
        }

    total_chunks = sum(len(c_list) for c_list in chunks_per_document)
    total_tokens = sum(c.token_count for c_list in chunks_per_document for c in c_list)

    avg_chunks_per_doc = total_chunks / num_docs
    avg_tokens_per_chunk = (total_tokens / total_chunks) if total_chunks > 0 else 0.0
    avg_latency = (sum(query_latencies_ms) / len(query_latencies_ms)) if query_latencies_ms else 0.0

    return {
        "chunks_per_doc": float(avg_chunks_per_doc),
        "tokens_per_chunk": float(avg_tokens_per_chunk),
        "build_time_sec": float(total_build_time_sec),
        "retrieval_latency_ms": float(avg_latency),
    }
