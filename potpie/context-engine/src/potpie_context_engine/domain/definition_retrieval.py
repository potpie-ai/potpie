"""Definition wording is a relevance signal, never corroboration."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

from potpie_context_core.definition_query import definition_subject
from potpie_context_core.graph_contract import TruthClass
from potpie_context_engine.domain.ranking import RankedItem

_UNCERTAIN = re.compile(
    r"\b(?:not|incorrect|wrong|false|rejected|might|may|could|perhaps|possibly|"
    r"guess|hypothetical|isn't|isn’t|doesn't|doesn’t)\b",
    re.IGNORECASE,
)
_AUTHORITATIVE = {
    TruthClass.authoritative_fact.value,
    TruthClass.source_observation.value,
}


def has_definition(text: str | None, subject: str | None) -> bool:
    """Match an affirmative definition of this exact term in one sentence.

    Deliberately conservative: a sentence questioning or rejecting an expansion
    stays retrievable but gets no affirmative-definition bonus.
    """
    if not subject or not text:
        return False
    term = re.escape(subject)
    pattern = re.compile(
        rf"\b{term}\b\s+(?:stands\s+for|means|is\s+(?:short\s+for|an?\s+"
        rf"(?:abbreviation|acronym)\s+for))\s+[\w]"
        rf"|\bfull\s+form\s+of\s+{term}\b\s+is\s+[\w]"
        rf"|\b{term}\b\s*[:=]\s*[\w]",
        re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if _UNCERTAIN.search(sentence) or "?" in sentence:
            continue
        if pattern.search(sentence) or _parenthetical_definition(sentence, subject):
            return True
    return False


def _parenthetical_definition(text: str, subject: str) -> bool:
    term = re.escape(subject)
    patterns = (
        rf"\b{term}\b\s*\(([^()\n]{{2,120}})\)",
        rf"((?:[^\W\d_]+[ /-]+){{1,11}}[^\W\d_]+)\s*\(\s*{term}\s*\)",
    )
    for index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            words = re.findall(r"[^\W\d_]+", match[1])
            # Before (PMS), a sentence can have prose preceding the expansion.
            if index == 1:
                words = words[-len(subject) :]
            if "".join(word[0] for word in words).casefold() == subject.casefold():
                return True
    return False


def rank_definitions(
    items: Sequence[RankedItem], query: str | None
) -> list[RankedItem]:
    """Rerank documentation before truncation, retaining original evidence fields.

    Authority requires both stored truth metadata and a source reference. It is
    not inferred from confident prose or from the presence of a DOCUMENTS edge.
    Even an authoritative source can be wrong; callers still get the source and
    the separate factors, not an invented confidence or corroboration count.
    """
    subject = definition_subject(query, allow_bare=True)
    if not subject:
        return list(items)
    ranked = []
    for item in items:
        payload = item.candidate.payload
        match = float(
            has_definition(payload.get("fact") or payload.get("snippet"), subject)
        )
        supported = float(
            bool(
                match
                and payload.get("truth") in _AUTHORITATIVE
                and (payload.get("source_refs") or payload.get("source_ref"))
                and item.candidate.strength in {"deterministic", "attested"}
            )
        )
        ranked.append(
            replace(
                item,
                score=0.35 * item.score + 0.4 * match + 0.25 * supported,
                breakdown={
                    **dict(item.breakdown),
                    "base_score": item.score,
                    "definition_match": match,
                    "definition_authority": supported,
                },
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)
