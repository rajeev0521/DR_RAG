"""Embedding extraction wrapper with unit-normalization and caching."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Tuple
import numpy as np

# Global model cache to avoid re-loading PyTorch weights on every arm/seed
_GLOBAL_MODEL_CACHE: Dict[Tuple[str, Optional[str]], object] = {}

# Global in-memory embedding cache: (model_name, sha256) -> np.ndarray
_GLOBAL_EMB_MEM_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


class EmbeddingModelWrapper:
    """Wrapper around SentenceTransformer models ensuring unit-normalized embeddings with disk/RAM caching."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_dir: Optional[str] = "data/.cache/embeddings",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            model_slug = "".join(c if c.isalnum() else "_" for c in self.model_name)
            self.db_path = self.cache_dir / f"emb_{model_slug}.sqlite"
            self._init_db()
        else:
            self.db_path = None

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
            )
            conn.commit()

    def _load_model(self):
        cache_key = (self.model_name, self.device)
        if cache_key in _GLOBAL_MODEL_CACHE:
            self._model = _GLOBAL_MODEL_CACHE[cache_key]
            return

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                _GLOBAL_MODEL_CACHE[cache_key] = self._model
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}'. "
                    f"Ensure sentence-transformers is installed. Underlying error: {exc}"
                )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Computes L2-normalized embeddings for a list of texts.
        Utilizes both an in-memory and SQLite-backed disk cache.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), normalized to unit norm.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        n = len(texts)
        text_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

        result_embeddings: List[Optional[np.ndarray]] = [None] * n
        missing_indices: List[int] = []

        # 1. Check in-memory cache
        for i, h in enumerate(text_hashes):
            mem_key = (self.model_name, h)
            if mem_key in _GLOBAL_EMB_MEM_CACHE:
                result_embeddings[i] = _GLOBAL_EMB_MEM_CACHE[mem_key]
            else:
                missing_indices.append(i)

        # 2. Check disk SQLite cache for remaining
        if missing_indices and self.db_path and self.db_path.exists():
            missing_hashes = [text_hashes[i] for i in missing_indices]
            # Query in chunks to avoid SQLite variable limits
            chunk_size = 500
            for start in range(0, len(missing_hashes), chunk_size):
                chunk_h = missing_hashes[start : start + chunk_size]
                placeholders = ",".join("?" * len(chunk_h))
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f"SELECT hash, dim, vec FROM embeddings WHERE hash IN ({placeholders})",
                        chunk_h,
                    )
                    rows = cursor.fetchall()
                    for r_hash, r_dim, r_vec in rows:
                        arr = np.frombuffer(r_vec, dtype=np.float32).copy()
                        _GLOBAL_EMB_MEM_CACHE[(self.model_name, r_hash)] = arr

            # Re-evaluate missing after disk check
            still_missing = []
            for idx in missing_indices:
                h = text_hashes[idx]
                mem_key = (self.model_name, h)
                if mem_key in _GLOBAL_EMB_MEM_CACHE:
                    result_embeddings[idx] = _GLOBAL_EMB_MEM_CACHE[mem_key]
                else:
                    still_missing.append(idx)
            missing_indices = still_missing

        # 3. Model inference for truly missing texts
        if missing_indices:
            self._load_model()
            texts_to_encode = [texts[i] for i in missing_indices]
            raw_embeddings = self._model.encode(
                texts_to_encode,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Explicit safety check for unit normalization
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = (raw_embeddings / norms).astype(np.float32)

            # Store in mem and prepare batch insert for SQLite
            db_records = []
            for j, idx in enumerate(missing_indices):
                vec = normalized[j]
                h = text_hashes[idx]
                _GLOBAL_EMB_MEM_CACHE[(self.model_name, h)] = vec
                result_embeddings[idx] = vec
                db_records.append((h, int(vec.shape[0]), vec.tobytes()))

            if self.db_path and db_records:
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany(
                        "INSERT OR IGNORE INTO embeddings (hash, dim, vec) VALUES (?, ?, ?)",
                        db_records,
                    )
                    conn.commit()

        return np.vstack(result_embeddings)

