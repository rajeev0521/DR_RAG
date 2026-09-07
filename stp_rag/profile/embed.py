"""Embedding extraction wrapper with unit-normalization and caching."""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class EmbeddingModelWrapper:
    """Wrapper around SentenceTransformer models ensuring unit-normalized embeddings."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}'. "
                    f"Ensure sentence-transformers is installed. Underlying error: {exc}"
                )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Computes L2-normalized embeddings for a list of texts.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), normalized to unit norm.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._load_model()
        raw_embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Built-in L2 unit normalization
        )

        # Explicit safety check for unit normalization
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (raw_embeddings / norms).astype(np.float32)
        return normalized
