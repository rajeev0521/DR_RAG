"""Dense retrieval stage over isolated vector store."""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np

from ..indexing.qdrant_store import STPVectorStore
from ..profile.embed import EmbeddingModelWrapper


class DenseRetriever:
    """Orchestrates query embedding and dense similarity search."""

    def __init__(
        self,
        vector_store: STPVectorStore,
        embedder: EmbeddingModelWrapper,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Tuple[List[int], float]:
        """
        Retrieves top_k chunks for a given query text.

        Returns:
            (retrieved_chunk_ids, retrieval_latency_ms)
        """
        # Embed query text
        query_vec = self.embedder.embed_texts([query])[0]
        # Search vector store
        return self.vector_store.search(query_vec, top_k=top_k)

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int = 10,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Batch retrieves top_k chunks for multiple queries.

        Returns:
            (list_of_retrieved_ids, list_of_latencies_ms)
        """
        all_ids = []
        all_latencies = []
        for q in queries:
            cids, lat = self.retrieve(q, top_k=top_k)
            all_ids.append(cids)
            all_latencies.append(lat)
        return all_ids, all_latencies
