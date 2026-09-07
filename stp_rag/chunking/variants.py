"""Ablation variants matching Table 1 of the STP-RAG paper."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..ingestion.structure_units import StructureUnit
from ..profile.stp import STPResult
from .segment import Chunk, segment_document

VALID_SYSTEMS = [
    "fixed_size",
    "similarity_threshold",
    "velocity_only",
    "velocity_acceleration",
    "full_stp",
    "full_stp_context",
]


def build_chunks_for_variant(
    variant: str,
    units: List[StructureUnit],
    stp: STPResult,
    embeddings: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
) -> List[Chunk]:
    """
    Constructs chunks for one of the six Table 1 ablation arms via a shared code path.
    """
    if variant not in VALID_SYSTEMS:
        raise ValueError(f"Unknown variant '{variant}'. Must be one of {VALID_SYSTEMS}")

    p = params or {}
    l_min = int(p.get("L_min", 100))
    l_max = int(p.get("L_max", 500))
    tau_b = float(p.get("tau_B", 0.5))

    # Base parameter dict
    w_v = float(p.get("w_v", 1.0))
    w_a = float(p.get("w_a", 0.5))
    w_sigma = float(p.get("w_sigma", 0.5))
    w_s = float(p.get("w_s", 0.8))
    b = float(p.get("b", -1.0))

    alpha = float(p.get("alpha", 0.5))
    beta = float(p.get("beta", 0.3))
    gamma = float(p.get("gamma", 0.2))

    if variant == "fixed_size":
        # Zero out all transition probabilities; force split at L_max
        boundary_weights = {"w_v": 0.0, "w_a": 0.0, "w_sigma": 0.0, "w_s": 0.0, "b": -999.0}
        budget_weights = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        return segment_document(
            units=units,
            stp=stp,
            embeddings=embeddings,
            tau_B=0.5,
            L_min=l_max,
            L_max=l_max,
            boundary_weights=boundary_weights,
            budget_weights=budget_weights,
        )

    elif variant == "similarity_threshold":
        # Segment solely on raw displacement d_i > tau_sim
        tau_sim = float(p.get("tau_sim", 0.35))
        chunks = []
        n = len(units)
        if n == 0:
            return []
        current_indices = [0]
        cur_len = units[0].token_count
        start_tok = 0
        chunk_id = 0

        for i in range(1, n):
            u_tok = units[i].token_count
            forced = (cur_len + u_tok >= l_max)
            dist_accepted = (stp.d[i] > tau_sim and cur_len >= l_min)

            if forced or dist_accepted:
                text = " ".join(units[idx].text for idx in current_indices)
                end_tok = units[current_indices[-1]].cumulative_tokens
                c_emb = None
                if embeddings is not None and len(embeddings) > 0:
                    c_emb = np.mean(embeddings[current_indices], axis=0)
                    norm = np.linalg.norm(c_emb)
                    if norm > 0:
                        c_emb = c_emb / norm

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        token_count=cur_len,
                        unit_ids=list(current_indices),
                        start_token=start_tok,
                        end_token=end_tok,
                        embedding=c_emb,
                        metadata={"forced_split": forced, "similarity_distance": float(stp.d[i])},
                    )
                )
                chunk_id += 1
                current_indices = [i]
                cur_len = u_tok
                start_tok = end_tok
            else:
                current_indices.append(i)
                cur_len += u_tok

        if current_indices:
            text = " ".join(units[idx].text for idx in current_indices)
            end_tok = units[current_indices[-1]].cumulative_tokens
            c_emb = None
            if embeddings is not None and len(embeddings) > 0:
                c_emb = np.mean(embeddings[current_indices], axis=0)
                norm = np.linalg.norm(c_emb)
                if norm > 0:
                    c_emb = c_emb / norm
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    token_count=cur_len,
                    unit_ids=list(current_indices),
                    start_token=start_tok,
                    end_token=end_tok,
                    embedding=c_emb,
                    metadata={"forced_split": False},
                )
            )
        return chunks

    elif variant == "velocity_only":
        # w_a = 0, w_sigma = 0, beta = 0, gamma = 0
        boundary_weights = {"w_v": w_v, "w_a": 0.0, "w_sigma": 0.0, "w_s": w_s, "b": b}
        budget_weights = {"alpha": alpha, "beta": 0.0, "gamma": 0.0}
        return segment_document(
            units=units,
            stp=stp,
            embeddings=embeddings,
            tau_B=tau_b,
            L_min=l_min,
            L_max=l_max,
            boundary_weights=boundary_weights,
            budget_weights=budget_weights,
        )

    elif variant == "velocity_acceleration":
        # w_sigma = 0, gamma = 0
        boundary_weights = {"w_v": w_v, "w_a": w_a, "w_sigma": 0.0, "w_s": w_s, "b": b}
        budget_weights = {"alpha": alpha, "beta": beta, "gamma": 0.0}
        return segment_document(
            units=units,
            stp=stp,
            embeddings=embeddings,
            tau_B=tau_b,
            L_min=l_min,
            L_max=l_max,
            boundary_weights=boundary_weights,
            budget_weights=budget_weights,
        )

    elif variant in ("full_stp", "full_stp_context"):
        # All signals active
        boundary_weights = {"w_v": w_v, "w_a": w_a, "w_sigma": w_sigma, "w_s": w_s, "b": b}
        budget_weights = {"alpha": alpha, "beta": beta, "gamma": gamma}
        return segment_document(
            units=units,
            stp=stp,
            embeddings=embeddings,
            tau_B=tau_b,
            L_min=l_min,
            L_max=l_max,
            boundary_weights=boundary_weights,
            budget_weights=budget_weights,
        )

    raise RuntimeError(f"Unhandled variant: {variant}")
