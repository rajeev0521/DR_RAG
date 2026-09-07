"""Profile subpackage: embeddings, kinematic transitions, and structural signals."""

from .embed import EmbeddingModelWrapper
from .stp import STPResult, compute_stp, fit_normalization_stats, apply_normalization
from .structural_features import compute_structural_score

__all__ = [
    "EmbeddingModelWrapper",
    "STPResult",
    "compute_stp",
    "fit_normalization_stats",
    "apply_normalization",
    "compute_structural_score",
]
