"""Adaptive target chunk budget policy."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from ..profile.stp import STPResult

DEFAULT_BUDGET_WEIGHTS = {
    "alpha": 0.5,
    "beta": 0.3,
    "gamma": 0.2,
    "L_min": 100,
    "L_max": 500,
}


def compute_target_budgets(
    stp: STPResult,
    weights: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes local target budget score T_i and target token length L_i.

    T_i = alpha * v_hat + beta * max(a_hat, 0) + gamma * sigma_hat
    L_i = clip(L_max - (L_max - L_min) * T_i, L_min, L_max)

    Returns:
        (T, L): arrays of length n.
    """
    w = {**DEFAULT_BUDGET_WEIGHTS, **(weights or {})}
    alpha = w.get("alpha", 0.0)
    beta = w.get("beta", 0.0)
    gamma = w.get("gamma", 0.0)
    l_min = float(w.get("L_min", 100))
    l_max = float(w.get("L_max", 500))

    pos_a = np.maximum(stp.a_hat, 0.0)
    T = alpha * stp.v_hat + beta * pos_a + gamma * stp.sigma_hat
    L = np.clip(l_max - (l_max - l_min) * T, l_min, l_max)

    return T.astype(np.float32), L.astype(np.float32)
