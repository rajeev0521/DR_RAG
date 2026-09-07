"""Calibrated logistic boundary probability policy."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from ..profile.stp import STPResult

DEFAULT_BOUNDARY_WEIGHTS = {
    "w_v": 1.0,
    "w_a": 0.5,
    "w_sigma": 0.5,
    "w_s": 0.8,
    "b": -1.0,
}


def compute_boundary_probabilities(
    stp: STPResult,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Computes boundary probability P(B_i = 1) for each unit:
    P(B_i=1) = sigmoid(w_v * v_hat + w_a * a_hat + w_sigma * sigma_hat + w_s * S + b)

    Returns:
        np.ndarray of length n with values in [0, 1].
    """
    w = {**DEFAULT_BOUNDARY_WEIGHTS, **(weights or {})}
    w_v = w.get("w_v", 0.0)
    w_a = w.get("w_a", 0.0)
    w_sigma = w.get("w_sigma", 0.0)
    w_s = w.get("w_s", 0.0)
    b = w.get("b", 0.0)

    logits = (
        w_v * stp.v_hat
        + w_a * stp.a_hat
        + w_sigma * stp.sigma_hat
        + w_s * stp.S
        + b
    )

    # Numerically stable sigmoid with bounded logits
    clipped_logits = np.clip(logits, -50.0, 50.0)
    probs = np.where(
        clipped_logits >= 0,
        1.0 / (1.0 + np.exp(-clipped_logits)),
        np.exp(clipped_logits) / (1.0 + np.exp(clipped_logits)),
    )
    return probs.astype(np.float32)
