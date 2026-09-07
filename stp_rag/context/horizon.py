"""Dynamic context horizon computation."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

DEFAULT_HORIZON_PARAMS = {
    "H_min": 1,
    "H_max": 4,
    "lambda_1": 0.5,
    "lambda_2": 0.5,
    "lambda_3": 1.0,
}


def compute_context_horizon(
    sigma_hat: float,
    a_hat: float,
    r_signal: float,
    params: Optional[Dict[str, float]] = None,
) -> int:
    """
    Computes integer context horizon H_i (number of preceding chunks):
    H_i = clip(H_min + lambda_1 * sigma_hat + lambda_2 * max(a_hat, 0) + lambda_3 * R_i, H_min, H_max)
    """
    p = {**DEFAULT_HORIZON_PARAMS, **(params or {})}
    h_min = int(p.get("H_min", 1))
    h_max = int(p.get("H_max", 4))
    l1 = float(p.get("lambda_1", 0.5))
    l2 = float(p.get("lambda_2", 0.5))
    l3 = float(p.get("lambda_3", 1.0))

    pos_a = max(a_hat, 0.0)
    raw_h = h_min + (l1 * sigma_hat) + (l2 * pos_a) + (l3 * r_signal)
    clipped = int(np.clip(np.round(raw_h), h_min, h_max))
    return clipped
