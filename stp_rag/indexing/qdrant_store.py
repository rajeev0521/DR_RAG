"""Qdrant vector store management with isolated per-arm collections."""

from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

from ..chunking.segment import Chunk


class STPVectorStore:
    """Manages indexing and vector search for an ablation arm."""

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int = 1024,
        url: Optional[str] = "http://localhost:6333",
        use_memory: bool = False,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.url = url
        self.use_memory = use_memory
        self._client = None
        self._chunks: List[Chunk] = []

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if self.use_memory:
                self._client = QdrantClient(":memory:")
            else:
                try:
                    self._client = QdrantClient(url=self.url, timeout=5)
                    # Ping to verify
                    self._client.get_collections()
                except Exception:
                    # Fallback to local in-memory store for development/testing
                    self._client = QdrantClient(":memory:")

            # Ensure collection exists
            collections = [c.name for c in self._client.get_collections().collections]
            if self.collection_name in collections:
                self._client.delete_collection(self.collection_name)

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        return self._client

    def index_chunks(self, chunks: List[Chunk]) -> float:
        """
        Indexes chunks into the arm's dedicated collection.

        Returns:
            build_time_sec (float)
        """
        from qdrant_client.models import PointStruct

        start_time = time.perf_counter()
        client = self._get_client()
        self._chunks = chunks

        points = []
        for c in chunks:
            if c.embedding is None:
                continue
            point = PointStruct(
                id=c.chunk_id,
                vector=c.embedding.tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "text": c.augmented_text,
                    "raw_text": c.text,
                    "token_count": c.token_count,
                    "metadata": c.metadata,
                },
            )
            points.append(point)

        # Batch upload
        batch_size = 64
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        build_time = time.perf_counter() - start_time
        return build_time

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> tuple[List[int], float]:
        """
        Searches top_k most similar chunks.

        Returns:
            (list_of_chunk_ids, latency_ms)
        """
        start_time = time.perf_counter()
        client = self._get_client()

        # Execute query using query_points API
        results = client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        retrieved_ids = [hit.id for hit in results.points]
        return retrieved_ids, latency_ms
