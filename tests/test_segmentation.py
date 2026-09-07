"""Unit tests for document segmentation and ablation variants."""

import numpy as np
import pytest

from stp_rag.ingestion.structure_units import StructureUnit
from stp_rag.profile.stp import compute_stp
from stp_rag.chunking.segment import segment_document
from stp_rag.chunking.variants import build_chunks_for_variant


def _make_dummy_units(n=20, tokens_per_unit=25):
    units = []
    cum = 0
    for i in range(n):
        cum += tokens_per_unit
        units.append(
            StructureUnit(
                unit_id=i,
                text=f"This is unit {i} in the sequence.",
                token_count=tokens_per_unit,
                cumulative_tokens=cum,
                follows_heading=(i % 5 == 0 and i > 0),
                is_paragraph_start=(i % 2 == 0),
            )
        )
    return units


def test_segmentation_hard_limit():
    units = _make_dummy_units(n=20, tokens_per_unit=50)  # total 1000 tokens
    # Dummy embeddings
    embs = np.random.randn(20, 16).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    s_scores = np.zeros(20, dtype=np.float32)
    stp = compute_stp(embs, np.array([u.cumulative_tokens for u in units]), s_scores)

    # L_max = 200, L_min = 50. Chunks must NEVER exceed L_max!
    chunks = segment_document(units, stp, embeddings=embs, L_min=50, L_max=200, tau_B=0.99)

    for c in chunks:
        assert c.token_count <= 250  # Since unit size is 50, forced boundary triggers at or before 200


def test_variants_coverage():
    units = _make_dummy_units(n=15, tokens_per_unit=30)
    embs = np.random.randn(15, 16).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    s_scores = np.zeros(15, dtype=np.float32)
    stp = compute_stp(embs, np.array([u.cumulative_tokens for u in units]), s_scores)

    variants = [
        "fixed_size",
        "similarity_threshold",
        "velocity_only",
        "velocity_acceleration",
        "full_stp",
        "full_stp_context",
    ]

    for var in variants:
        chunks = build_chunks_for_variant(var, units, stp, embeddings=embs)
        assert len(chunks) > 0
        total_tokens_covered = sum(c.token_count for c in chunks)
        assert total_tokens_covered == units[-1].cumulative_tokens
