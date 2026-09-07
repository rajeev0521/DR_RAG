"""Unit test demonstrating adaptive chunk budget T_i / L_i modulates chunk length distributions."""

import numpy as np
import pytest

from stp_rag.chunking.variants import build_chunks_for_variant
from stp_rag.ingestion.structure_units import StructureUnit
from stp_rag.profile.stp import STPResult


def test_adaptive_budget_modulates_chunk_lengths():
    """
    Shows that two documents with identical token counts and identical L_min/L_max
    produce measurably different chunk-length distributions in full_stp based on
    their kinematic volatility profile.
    """
    n_units = 30
    unit_tok = 20  # 30 * 20 = 600 total tokens

    units = [
        StructureUnit(
            unit_id=i,
            text=f"Sentence {i} containing standard informational content for evaluation.",
            token_count=unit_tok,
            cumulative_tokens=(i + 1) * unit_tok,
        )
        for i in range(n_units)
    ]

    # Doc A: Stable region (low velocity, low acceleration, low volatility)
    # T_i -> 0  => Target length L_i -> L_max (200)
    stp_stable = STPResult(
        d=np.zeros(n_units, dtype=np.float32),
        delta_p=np.full(n_units, 1.0 / n_units, dtype=np.float32),
        v=np.zeros(n_units, dtype=np.float32),
        a=np.zeros(n_units, dtype=np.float32),
        sigma=np.zeros(n_units, dtype=np.float32),
        v_hat=np.full(n_units, -1.0, dtype=np.float32),
        a_hat=np.full(n_units, -1.0, dtype=np.float32),
        sigma_hat=np.full(n_units, -1.0, dtype=np.float32),
        S=np.zeros(n_units, dtype=np.float32),
    )

    # Doc B: Highly volatile / shifting region (high velocity, high acceleration, high volatility)
    # T_i -> large => Target length L_i -> L_min (60)
    stp_volatile = STPResult(
        d=np.ones(n_units, dtype=np.float32),
        delta_p=np.full(n_units, 1.0 / n_units, dtype=np.float32),
        v=np.full(n_units, 5.0, dtype=np.float32),
        a=np.full(n_units, 3.0, dtype=np.float32),
        sigma=np.full(n_units, 4.0, dtype=np.float32),
        v_hat=np.full(n_units, 3.0, dtype=np.float32),
        a_hat=np.full(n_units, 2.0, dtype=np.float32),
        sigma_hat=np.full(n_units, 3.0, dtype=np.float32),
        S=np.zeros(n_units, dtype=np.float32),
    )

    params = {
        "L_min": 60,
        "L_max": 200,
        "tau_B": 0.99,  # Soft threshold high so splits are driven by budget ceiling
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 0.2,
    }

    chunks_stable = build_chunks_for_variant("full_stp", units, stp_stable, params=params)
    chunks_volatile = build_chunks_for_variant("full_stp", units, stp_volatile, params=params)

    # The volatile document should produce more chunks (finer granularity) than the stable one
    assert len(chunks_volatile) > len(chunks_stable), (
        f"Volatile doc produced {len(chunks_volatile)} chunks; "
        f"stable doc produced {len(chunks_stable)} chunks."
    )

    # Average tokens per chunk should be significantly higher in the stable document
    mean_len_stable = np.mean([c.token_count for c in chunks_stable])
    mean_len_volatile = np.mean([c.token_count for c in chunks_volatile])

    assert mean_len_stable > mean_len_volatile, (
        f"Stable mean chunk len ({mean_len_stable}) should be greater than "
        f"volatile mean chunk len ({mean_len_volatile})."
    )
    assert mean_len_stable >= 140.0
    assert mean_len_volatile <= 70.0
