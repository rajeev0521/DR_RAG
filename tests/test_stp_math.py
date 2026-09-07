"""Unit tests for STP mathematical formulation and properties."""

import numpy as np
import pytest

from stp_rag.profile.stp import (
    compute_raw_kinematics,
    fit_normalization_stats,
    apply_normalization,
    compute_stp,
)


def test_kinematics_simple_vectors():
    # 3 orthogonal vectors (90 deg apart => cos = 0 => d = 1.0)
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    cumulative_tokens = np.array([10, 20, 30], dtype=np.int32)
    # Total tokens = 30
    # delta_p: [10/30, 10/30, 10/30] = [1/3, 1/3, 1/3]
    # d: [0.0, 1.0, 1.0]
    # v: [0.0, 1.0 / (1/3 + 1e-6) ~ 3.0, 1.0 / (1/3 + 1e-6) ~ 3.0]
    # a: [0.0, 0.0, v[2] - v[1] ~ 0.0]

    d, delta_p, v, a, sigma = compute_raw_kinematics(
        embeddings, cumulative_tokens, window_w=3
    )

    assert len(d) == 3
    assert d[0] == 0.0
    assert np.isclose(d[1], 1.0, atol=1e-5)
    assert np.isclose(d[2], 1.0, atol=1e-5)

    assert np.isclose(delta_p[0], 1.0 / 3.0, atol=1e-5)
    assert np.isclose(delta_p[1], 1.0 / 3.0, atol=1e-5)
    assert np.isclose(delta_p[2], 1.0 / 3.0, atol=1e-5)

    assert v[0] == 0.0
    assert np.isclose(v[1], 3.0, atol=1e-3)
    assert np.isclose(v[2], 3.0, atol=1e-3)

    # Acceleration at unit 3 (index 2) is v[2] - v[1] ~ 0.0
    assert np.isclose(a[2], 0.0, atol=1e-3)


def test_acceleration_sign_asymmetry():
    # Accelerating scenario: v increases from 1 to 5
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.43588989],  # Small distance
        [0.0, 1.0],         # Large distance
        [-1.0, 0.0],        # Huge distance
    ], dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    cumulative_tokens = np.array([10, 20, 30, 40], dtype=np.int32)
    d, delta_p, v, a, sigma = compute_raw_kinematics(embeddings, cumulative_tokens, window_w=3)

    # v[2] should be greater than v[1], so a[2] > 0
    assert a[2] > 0


def test_robust_normalization_frozen():
    v_train = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # outlier present
    stats = fit_normalization_stats(v_train, v_train, v_train)

    assert stats["v_median"] == 3.0
    # IQR is unaffected by extreme outlier 100.0
    assert stats["v_iqr"] == 2.0

    # Normalization of a test sample using frozen train stats
    x_test = np.array([3.0, 5.0])
    normed = apply_normalization(x_test, stats["v_median"], stats["v_iqr"])

    assert np.isclose(normed[0], 0.0, atol=1e-5)   # median maps to 0
    assert np.isclose(normed[1], 1.0, atol=1e-5)   # (5 - 3) / 2 = 1.0
