"""Query routing interface for single-hop vs. multi-hop evidence selection."""

from __future__ import annotations

import re
from typing import Optional


class QueryRouter:
    """Classifies incoming questions to calibrate retrieval depth and hop routing."""

    def __init__(self, model_checkpoint: Optional[str] = None):
        self.model_checkpoint = model_checkpoint

    def route(self, query: str) -> str:
        """
        Determines whether the query is 'single_hop' or 'multi_hop'.

        Uses pattern heuristics or classifier checkpoint.
        Multi-hop questions typically combine conjunctions, comparison words,
        or multiple entity references (e.g. 'both X and Y', 'which ... also ...').
        """
        multi_hop_patterns = [
            r"\b(both|and|as well as)\b.*\b(who|which|what|where)\b",
            r"\b(compared to|in comparison with|between .* and)\b",
            r"\bwhich\b.*\b(first|second|earlier|later|more|less)\b",
            r"\b(after|before|during)\b.*\b(did|was|were)\b",
        ]
        for pattern in multi_hop_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return "multi_hop"

        return "single_hop"
