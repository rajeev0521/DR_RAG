"""Semantic Transition Profile (STP) mathematical engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

EPSILON = 1e-6
DELTA = 1e-6


@dataclass
class STPResult:
    """Encapsulates raw and normalized kinematic transition signals."""
    d: np.ndarray          # Semantic displacement d_i
    delta_p: np.ndarray    # Document-relative progression Δp_i
    v: np.ndarray          # Semantic velocity v_i
    a: np.ndarray          # Semantic acceleration a_i
    sigma: np.ndarray      # Semantic volatility σ_i
    v_hat: np.ndarray      # Robustly normalized velocity v̂_i
    a_hat: np.ndarray      # Robustly normalized acceleration â_i
    sigma_hat: np.ndarray  # Robustly normalized volatility σ̂_i
    S: np.ndarray          # Scaled structural score S_i

    @property
    def profile_matrix(self) -> np.ndarray:
        """Returns the n x 4 STP matrix [v_hat, a_hat, sigma_hat, S]."""
        return np.column_stack([self.v_hat, self.a_hat, self.sigma_hat, self.S])


def compute_raw_kinematics(
    embeddings: np.ndarray,
    cumulative_tokens: np.ndarray,
    window_w: int = 5,
    epsilon: float = EPSILON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes raw kinematics (d, Δp, v, a, σ) over ordered document units.

    Args:
        embeddings: (n, d) normalized embeddings matrix.
        cumulative_tokens: (n,) array of cumulative token positions p_i.
        window_w: Trailing window size for volatility.
        epsilon: Small constant to prevent division by zero.

    Returns:
        (d, delta_p, v, a, sigma) arrays each of length n.
    """
    n = len(embeddings)
    if n == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, empty

    total_n_tokens = float(cumulative_tokens[-1]) if cumulative_tokens[-1] > 0 else 1.0

    # 1. Progression Δp_i = (p_i - p_{i-1}) / N
    delta_p = np.zeros(n, dtype=np.float32)
    delta_p[0] = cumulative_tokens[0] / total_n_tokens
    if n > 1:
        delta_p[1:] = np.diff(cumulative_tokens) / total_n_tokens

    # 2. Displacement d_i = 1 - cos(e_{i-1}, e_i)
    d = np.zeros(n, dtype=np.float32)
    if n > 1:
        # Since embeddings are unit-normalized, cos(e1, e2) is dot product
        cos_sim = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        d[1:] = 1.0 - cos_sim

    # 3. Velocity v_i = d_i / (Δp_i + ε)
    v = np.zeros(n, dtype=np.float32)
    if n > 1:
        v[1:] = d[1:] / (delta_p[1:] + epsilon)

    # 4. Acceleration a_i = v_i - v_{i-1} for i >= 3
    a = np.zeros(n, dtype=np.float32)
    if n >= 3:
        # a[2] corresponds to unit index 2 (i=3 1-indexed): v[2] - v[1]
        a[2:] = v[2:] - v[1:-1]

    # 5. Volatility σ_i = std(v) over trailing window w
    sigma = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start_idx = max(0, i - window_w + 1)
        window_slice = v[start_idx : i + 1]
        if len(window_slice) > 1:
            sigma[i] = np.std(window_slice)
        else:
            sigma[i] = 0.0

    return d, delta_p, v, a, sigma


def fit_normalization_stats(
    v_all: np.ndarray,
    a_all: np.ndarray,
    sigma_all: np.ndarray,
) -> Dict[str, float]:
    """
    Fits robust normalization statistics (median, IQR) strictly on dev-split data.
    """
    def _calc_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(arr) == 0:
            return {f"{prefix}_median": 0.0, f"{prefix}_iqr": 1.0}
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        iqr = float(q75 - q25)
        return {
            f"{prefix}_median": float(q50),
            f"{prefix}_iqr": float(iqr if iqr > 1e-8 else 1.0),
        }

    stats = {}
    stats.update(_calc_stats(v_all, "v"))
    stats.update(_calc_stats(a_all, "a"))
    stats.update(_calc_stats(sigma_all, "sigma"))
    return stats


def apply_normalization(
    x: np.ndarray,
    median_val: float,
    iqr_val: float,
    delta: float = DELTA,
) -> np.ndarray:
    """
    Applies frozen robust normalization: x̂ = (x - median) / (IQR + δ).
    """
    return ((x - median_val) / (iqr_val + delta)).astype(np.float32)


def compute_stp(
    embeddings: np.ndarray,
    cumulative_tokens: np.ndarray,
    structural_score: np.ndarray,
    norm_stats: Optional[Dict[str, float]] = None,
    window_w: int = 5,
) -> STPResult:
    """
    Calculates full STPResult for a document.

    If norm_stats is None, computes self-normalized statistics (dev mode only).
    """
    d, delta_p, v, a, sigma = compute_raw_kinematics(
        embeddings=embeddings,
        cumulative_tokens=cumulative_tokens,
        window_w=window_w,
    )

    if norm_stats is None:
        norm_stats = fit_normalization_stats(v, a, sigma)

    v_hat = apply_normalization(v, norm_stats["v_median"], norm_stats["v_iqr"])
    a_hat = apply_normalization(a, norm_stats["a_median"], norm_stats["a_iqr"])
    sigma_hat = apply_normalization(sigma, norm_stats["sigma_median"], norm_stats["sigma_iqr"])

    return STPResult(
        d=d,
        delta_p=delta_p,
        v=v,
        a=a,
        sigma=sigma,
        v_hat=v_hat,
        a_hat=a_hat,
        sigma_hat=sigma_hat,
        S=structural_score.astype(np.float32),
    )
