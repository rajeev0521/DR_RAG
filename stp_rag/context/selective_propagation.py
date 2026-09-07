"""Selective context candidate ranking and propagation."""

from __future__ import annotations

import re
from typing import Dict, List, Optional
import numpy as np

from ..chunking.segment import Chunk
from ..profile.stp import STPResult
from .horizon import compute_context_horizon
from .reference_signal import compute_reference_signal, extract_abbreviations

DEFAULT_RANKING_PARAMS = {
    "mu_1": 0.5,
    "mu_2": 0.3,
    "mu_3": 0.5,
    "tau_R": 0.4,
}


def _token_overlap(text_a: str, text_b: str) -> float:
    """Computes Jaccard overlap of distinctive words (nouns/named tokens)."""
    words_a = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", text_a.lower()))
    words_b = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)
    return len(intersection) / len(union)


def _check_ref_resolve(cand_text: str, curr_text: str) -> float:
    """Checks if predecessor candidate contains definitions for terms in current chunk."""
    curr_abbrevs = extract_abbreviations(curr_text)
    for abbr in curr_abbrevs:
        if len(abbr) >= 2 and abbr in cand_text:
            return 1.0
    return 0.0


def rank_candidate_relevance(
    cand_chunk: Chunk,
    curr_chunk: Chunk,
    mu_1: float = 0.5,
    mu_2: float = 0.3,
    mu_3: float = 0.5,
) -> float:
    """
    Computes Rel(c_j, c_i) = mu_1 * cos(e_j, e_i) + mu_2 * DepMatch(c_j, c_i) + mu_3 * RefResolve(c_j, c_i).
    """
    # 1. Cosine similarity
    sim = 0.0
    if cand_chunk.embedding is not None and curr_chunk.embedding is not None:
        sim = float(np.dot(cand_chunk.embedding, curr_chunk.embedding))

    # 2. Dependency / entity overlap match
    dep_match = _token_overlap(cand_chunk.text, curr_chunk.text)

    # 3. Reference resolution match
    ref_resolve = _check_ref_resolve(cand_chunk.text, curr_chunk.text)

    rel_score = (mu_1 * sim) + (mu_2 * dep_match) + (mu_3 * ref_resolve)
    return float(rel_score)


def select_and_propagate_context(
    chunks: List[Chunk],
    stp: STPResult,
    horizon_params: Optional[Dict] = None,
    ranking_params: Optional[Dict] = None,
) -> List[Chunk]:
    """
    Executes selective context propagation across segmented chunks.
    Augments chunks in-place with selected predecessor context.
    """
    if not chunks:
        return []

    r_params = {**DEFAULT_RANKING_PARAMS, **(ranking_params or {})}
    mu_1 = float(r_params.get("mu_1", 0.5))
    mu_2 = float(r_params.get("mu_2", 0.3))
    mu_3 = float(r_params.get("mu_3", 0.5))
    tau_r = float(r_params.get("tau_R", 0.4))

    for i in range(len(chunks)):
        c_i = chunks[i]
        # Identify representative unit index for chunk
        lead_unit_idx = c_i.unit_ids[0] if c_i.unit_ids else 0

        sigma_val = float(stp.sigma_hat[lead_unit_idx]) if lead_unit_idx < len(stp.sigma_hat) else 0.0
        a_val = float(stp.a_hat[lead_unit_idx]) if lead_unit_idx < len(stp.a_hat) else 0.0
        r_sig = compute_reference_signal(c_i.text)

        h_i = compute_context_horizon(
            sigma_hat=sigma_val,
            a_hat=a_val,
            r_signal=r_sig,
            params=horizon_params,
        )

        start_j = max(0, i - h_i)
        selected = []
        for j in range(start_j, i):
            cand = chunks[j]
            rel = rank_candidate_relevance(cand, c_i, mu_1=mu_1, mu_2=mu_2, mu_3=mu_3)
            if rel > tau_r:
                selected.append(cand)

        c_i.selected_context = selected
        c_i.metadata["context_horizon"] = h_i
        c_i.metadata["reference_signal"] = r_sig
        c_i.metadata["selected_predecessor_ids"] = [p.chunk_id for p in selected]

    return chunks
