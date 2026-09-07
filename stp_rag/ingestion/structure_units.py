"""Structure-aware unit extraction and metadata tagging."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

CLAUSE_MARKERS = {
    "however", "except", "provided that", "notwithstanding", "moreover",
    "furthermore", "nevertheless", "nonetheless", "conversely", "whereas",
    "in contrast", "alternatively", "on the other hand", "that said",
}

CLAUSE_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(marker) for marker in CLAUSE_MARKERS) + r")\b",
    re.IGNORECASE,
)

HEADING_REGEX = re.compile(r"^(?:#{1,6}\s+|[A-Z0-9\.\s]{3,}:?$)", re.MULTILINE)
LIST_ITEM_REGEX = re.compile(r"^(\s*[\*\-\+]\s+|\s*\d+[\.\)]\s+)")


@dataclass
class StructureUnit:
    """Represents a discrete, ordered structural unit of a document."""
    unit_id: int
    text: str
    token_count: int
    cumulative_tokens: int  # p_i: cumulative token count up to and including u_i
    follows_heading: bool = False
    follows_list_item: bool = False
    is_paragraph_start: bool = False
    follows_clause_marker: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def s_tuple(self) -> tuple[int, int, int, int]:
        """Binary tuple representation of structural cues."""
        return (
            int(self.follows_heading),
            int(self.follows_list_item),
            int(self.is_paragraph_start),
            int(self.follows_clause_marker),
        )


def _tokenize(text: str) -> list[str]:
    """Lightweight whitespace and punctuation tokenizer."""
    return re.findall(r"\b\w+\b|[^\w\s]", text)


def extract_structure_aware_units(
    text: str,
    granularity: str = "sentence",
) -> List[StructureUnit]:
    """
    Extracts structure-aware units from a document.

    Args:
        text: Raw document text (plain text or markdown).
        granularity: 'sentence' or 'paragraph' (must be constant per corpus).

    Returns:
        List of StructureUnit instances with cumulative token counts and structural cues.
    """
    lines = text.splitlines()
    raw_blocks: list[dict] = []

    last_was_heading = False
    last_was_list_item = False

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        is_heading = bool(HEADING_REGEX.match(stripped))
        is_list = bool(LIST_ITEM_REGEX.match(line))

        raw_blocks.append({
            "line_idx": line_idx,
            "text": stripped,
            "is_heading": is_heading,
            "is_list": is_list,
            "follows_heading": last_was_heading,
            "follows_list_item": last_was_list_item,
        })

        last_was_heading = is_heading
        last_was_list_item = is_list

    units: List[StructureUnit] = []
    cumulative_tokens = 0
    unit_id = 0

    if granularity == "paragraph":
        # Group by double-newline separated blocks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for p_idx, p_text in enumerate(paragraphs):
            tokens = _tokenize(p_text)
            token_count = len(tokens)
            cumulative_tokens += token_count

            has_clause = bool(CLAUSE_REGEX.search(p_text[:80]))
            is_heading = bool(HEADING_REGEX.match(p_text))
            is_list = bool(LIST_ITEM_REGEX.match(p_text))

            units.append(
                StructureUnit(
                    unit_id=unit_id,
                    text=p_text,
                    token_count=token_count,
                    cumulative_tokens=cumulative_tokens,
                    follows_heading=p_idx > 0 and bool(HEADING_REGEX.match(paragraphs[p_idx - 1])),
                    follows_list_item=p_idx > 0 and bool(LIST_ITEM_REGEX.match(paragraphs[p_idx - 1])),
                    is_paragraph_start=True,
                    follows_clause_marker=has_clause,
                )
            )
            unit_id += 1
        return units

    # Sentence granularity (default)
    # Split paragraphs first to accurately tag is_paragraph_start
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    prev_sentence_ended_heading = False
    prev_sentence_ended_list = False

    for p_text in paragraphs:
        # Regex sentence boundary detection
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", p_text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        for s_idx, s_text in enumerate(sentences):
            tokens = _tokenize(s_text)
            token_count = max(len(tokens), 1)
            cumulative_tokens += token_count

            is_start_of_para = (s_idx == 0)
            has_clause = bool(CLAUSE_REGEX.search(s_text[:50]))
            is_heading = bool(HEADING_REGEX.match(s_text))
            is_list = bool(LIST_ITEM_REGEX.match(s_text))

            follows_head = prev_sentence_ended_heading if is_start_of_para else False
            follows_list = prev_sentence_ended_list if is_start_of_para else False

            unit = StructureUnit(
                unit_id=unit_id,
                text=s_text,
                token_count=token_count,
                cumulative_tokens=cumulative_tokens,
                follows_heading=follows_head,
                follows_list_item=follows_list,
                is_paragraph_start=is_start_of_para,
                follows_clause_marker=has_clause,
            )
            units.append(unit)
            unit_id += 1

            prev_sentence_ended_heading = is_heading
            prev_sentence_ended_list = is_list

    return units
