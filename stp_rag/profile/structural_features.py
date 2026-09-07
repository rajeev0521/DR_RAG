"""Structural features scoring and scaling."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..ingestion.structure_units import StructureUnit

DEFAULT_ETA = {
    "eta_heading": 1.0,
    "eta_list": 0.5,
    "eta_paragraph": 0.8,
    "eta_clause": 0.6,
}


def compute_structural_score(
    units: List[StructureUnit],
    eta: Optional[Dict[str, float]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> tuple[np.ndarray, float, float]:
    """
    Computes structural boundary score S_i across units.

    S_i = eta_1 * heading + eta_2 * list + eta_3 * paragraph - eta_4 * clause

    Args:
        units: List of StructureUnit objects.
        eta: Coefficients dictionary (eta_heading, eta_list, eta_paragraph, eta_clause).
        min_val: Min value for scaling (if frozen from dev).
        max_val: Max value for scaling (if frozen from dev).

    Returns:
        (S_scaled, observed_min, observed_max)
    """
    if not units:
        return np.empty(0, dtype=np.float32), 0.0, 1.0

    params = {**DEFAULT_ETA, **(eta or {})}
    e_head = params["eta_heading"]
    e_list = params["eta_list"]
    e_para = params["eta_paragraph"]
    e_clause = params["eta_clause"]

    raw_scores = np.zeros(len(units), dtype=np.float32)

    for i, u in enumerate(units):
        raw = (
            e_head * float(u.follows_heading)
            + e_list * float(u.follows_list_item)
            + e_para * float(u.is_paragraph_start)
            - e_clause * float(u.follows_clause_marker)
        )
        raw_scores[i] = raw

    obs_min = float(np.min(raw_scores))
    obs_max = float(np.max(raw_scores))

    scale_min = min_val if min_val is not None else obs_min
    scale_max = max_val if max_val is not None else obs_max

    range_val = scale_max - scale_min
    if abs(range_val) < 1e-8:
        scaled = np.zeros_like(raw_scores)
    else:
        scaled = np.clip((raw_scores - scale_min) / range_val, 0.0, 1.0)

    return scaled, obs_min, obs_max
