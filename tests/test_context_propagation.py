"""Unit tests for reference signals, context horizons, and selective propagation."""

import numpy as np
import pytest

from stp_rag.context.horizon import compute_context_horizon
from stp_rag.context.reference_signal import compute_reference_signal
from stp_rag.chunking.segment import Chunk
from stp_rag.profile.stp import STPResult
from stp_rag.context.selective_propagation import select_and_propagate_context


def test_reference_signal_detection():
    # Anaphoric demonstrative
    text1 = "In this condition, the threshold shall not exceed 10%."
    assert compute_reference_signal(text1) == 1.0

    # Discourse connective
    text2 = "Therefore, the hypothesis is confirmed."
    assert compute_reference_signal(text2) == 1.0

    # Independent statement without triggers
    text3 = "Water boils at 100 degrees Celsius at sea level."
    assert compute_reference_signal(text3) == 0.0


def test_horizon_bounds():
    # Extreme high values must clip at H_max
    h_max = compute_context_horizon(sigma_hat=10.0, a_hat=10.0, r_signal=1.0)
    assert h_max == 4

    # Low/negative values must clip at H_min
    h_min = compute_context_horizon(sigma_hat=-5.0, a_hat=-5.0, r_signal=0.0)
    assert h_min == 1


def test_selective_propagation_augmentation():
    c0 = Chunk(chunk_id=0, text="Definition: X is a semantic control parameter.", token_count=10, unit_ids=[0], start_token=0, end_token=10)
    c1 = Chunk(chunk_id=1, text="Therefore, this condition applies directly to X.", token_count=10, unit_ids=[1], start_token=10, end_token=20)

    # Embeddings that are similar
    c0.embedding = np.array([1.0, 0.0], dtype=np.float32)
    c1.embedding = np.array([0.9, 0.4358], dtype=np.float32)

    stp = STPResult(
        d=np.zeros(2), delta_p=np.zeros(2), v=np.zeros(2), a=np.zeros(2), sigma=np.zeros(2),
        v_hat=np.zeros(2), a_hat=np.zeros(2), sigma_hat=np.zeros(2), S=np.zeros(2)
    )

    chunks = select_and_propagate_context([c0, c1], stp)
    assert len(chunks[1].selected_context) == 1
    assert chunks[1].selected_context[0].chunk_id == 0
    assert "Definition: X" in chunks[1].augmented_text
