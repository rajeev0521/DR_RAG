"""Chunking subpackage: boundary policies, adaptive budgets, and segmentation."""

from .boundary_policy import compute_boundary_probabilities
from .budget_policy import compute_target_budgets
from .segment import Chunk, segment_document
from .variants import build_chunks_for_variant

__all__ = [
    "compute_boundary_probabilities",
    "compute_target_budgets",
    "Chunk",
    "segment_document",
    "build_chunks_for_variant",
]
