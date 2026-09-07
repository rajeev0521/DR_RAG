"""Context propagation subpackage: reference detection, horizons, and selective augmentation."""

from .horizon import compute_context_horizon
from .reference_signal import compute_reference_signal
from .selective_propagation import select_and_propagate_context

__all__ = [
    "compute_context_horizon",
    "compute_reference_signal",
    "select_and_propagate_context",
]
