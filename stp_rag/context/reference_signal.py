"""Reference and dependency signal detector for chunks."""

from __future__ import annotations

import re
from typing import Set

ANAPHORIC_PATTERNS = [
    r"\b(this|these|that|those)\s+(?:condition|provision|section|requirement|rule|exception|result|finding|statement|method)\b",
    r"\b(the\s+above|the\s+aforementioned|the\s+latter|the\s+former)\b",
    r"^(?:It|They|These|Those)\s+(?:is|are|was|were|has|have)\b",
]

ANAPHORIC_REGEX = re.compile("|".join(ANAPHORIC_PATTERNS), re.IGNORECASE)

DISCOURSE_CONNECTIVES = {
    "therefore", "as a result", "similarly", "in contrast", "consequently",
    "accordingly", "thus", "hence", "furthermore", "moreover", "in addition",
    "specifically", "for example", "in other words",
}

DISCOURSE_REGEX = re.compile(
    r"^\s*(" + "|".join(re.escape(c) for c in DISCOURSE_CONNECTIVES) + r")\b",
    re.IGNORECASE,
)

ABBREVIATION_REGEX = re.compile(r"\b[A-Z]{2,6}\b")


def extract_abbreviations(text: str) -> Set[str]:
    """Finds uppercase acronyms/abbreviations."""
    return set(ABBREVIATION_REGEX.findall(text))


def compute_reference_signal(chunk_text: str) -> float:
    """
    Computes R_i = max(R_coref, R_abbrev, R_lexcue) in [0, 1].

    - R_coref = 1 if chunk contains unresolved demonstrative / anaphoric phrase.
    - R_abbrev = 1 if chunk uses acronyms without local definition in parentheses.
    - R_lexcue = 1 if chunk starts with discourse connective.
    """
    # 1. Coreference / anaphoric cue
    r_coref = 1.0 if bool(ANAPHORIC_REGEX.search(chunk_text)) else 0.0

    # 2. Discourse connective cue
    r_lexcue = 1.0 if bool(DISCOURSE_REGEX.search(chunk_text)) else 0.0

    # 3. Abbreviation cue (acronym used without definition like "Short Name (SN)")
    r_abbrev = 0.0
    abbrevs = extract_abbreviations(chunk_text)
    for abbr in abbrevs:
        if abbr in {"THE", "AND", "FOR", "NOT", "IEEE", "RAG", "STP"}:
            continue
        # Check if definition pattern "Full Name (ABBR)" is absent
        def_pattern = rf"\([\"']?{abbr}[\"']?\)"
        if not re.search(def_pattern, chunk_text):
            r_abbrev = 1.0
            break

    return float(max(r_coref, r_abbrev, r_lexcue))
