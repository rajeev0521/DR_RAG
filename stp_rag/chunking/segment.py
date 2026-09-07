"""Segmentation algorithm enforcing length boundaries and dynamic thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..ingestion.structure_units import StructureUnit
from ..profile.stp import STPResult
from .boundary_policy import compute_boundary_probabilities
from .budget_policy import compute_target_budgets


@dataclass
class Chunk:
    """Represents a coherent segment of text produced by chunking."""
    chunk_id: int
    text: str
    token_count: int
    unit_ids: List[int]
    start_token: int
    end_token: int
    embedding: Optional[np.ndarray] = None
    selected_context: List[Chunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def augmented_text(self) -> str:
        """Returns the indexed representation c̃_i = c_i || SelectedContext_i."""
        if not self.selected_context:
            return self.text
        context_str = "\n".join(c.text for c in self.selected_context)
        return f"{context_str}\n\n{self.text}"


def segment_document(
    units: List[StructureUnit],
    stp: STPResult,
    embeddings: Optional[np.ndarray] = None,
    tau_B: float = 0.5,
    L_min: int = 100,
    L_max: int = 500,
    boundary_weights: Optional[Dict[str, float]] = None,
    budget_weights: Optional[Dict[str, float]] = None,
) -> List[Chunk]:
    """
    Executes Algorithm A segmentation over ordered document units.

    Enforces:
    - Hard ceiling L_max forcing a split.
    - Soft boundary when P(B_i=1) > tau_B and current_length >= L_min.
    """
    n = len(units)
    if n == 0:
        return []

    probs = compute_boundary_probabilities(stp, boundary_weights)
    _, target_lengths = compute_target_budgets(
        stp,
        {**(budget_weights or {}), "L_min": L_min, "L_max": L_max},
    )

    chunks: List[Chunk] = []
    current_unit_indices: List[int] = [0]
    current_length = units[0].token_count
    start_tok = 0
    chunk_id = 0

    for i in range(1, n):
        u = units[i]
        u_tokens = u.token_count

        # Check conditions
        prob = float(probs[i])
        forced = (current_length + u_tokens >= L_max)
        prob_accepted = (prob > tau_B and current_length >= L_min)

        if forced or prob_accepted:
            # Finalize current chunk before unit i (or including unit i if forced on single large unit)
            chunk_text = " ".join(units[idx].text for idx in current_unit_indices)
            end_tok = units[current_unit_indices[-1]].cumulative_tokens

            # Compute mean embedding of chunk units if provided
            c_emb = None
            if embeddings is not None and len(embeddings) > 0:
                c_emb = np.mean(embeddings[current_unit_indices], axis=0)
                norm = np.linalg.norm(c_emb)
                if norm > 0:
                    c_emb = c_emb / norm

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    token_count=current_length,
                    unit_ids=list(current_unit_indices),
                    start_token=start_tok,
                    end_token=end_tok,
                    embedding=c_emb,
                    metadata={"forced_split": forced, "boundary_prob": prob},
                )
            )
            chunk_id += 1

            # Reset
            current_unit_indices = [i]
            current_length = u_tokens
            start_tok = end_tok
        else:
            current_unit_indices.append(i)
            current_length += u_tokens

    # Close remaining units as final chunk
    if current_unit_indices:
        chunk_text = " ".join(units[idx].text for idx in current_unit_indices)
        end_tok = units[current_unit_indices[-1]].cumulative_tokens
        c_emb = None
        if embeddings is not None and len(embeddings) > 0:
            c_emb = np.mean(embeddings[current_unit_indices], axis=0)
            norm = np.linalg.norm(c_emb)
            if norm > 0:
                c_emb = c_emb / norm

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                token_count=current_length,
                unit_ids=list(current_unit_indices),
                start_token=start_tok,
                end_token=end_tok,
                embedding=c_emb,
                metadata={"forced_split": False, "boundary_prob": 1.0},
            )
        )

    return chunks
