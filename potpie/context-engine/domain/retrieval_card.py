"""The retrieval card: the text a claim is embedded as (R2).

Recall quality comes from the *input*, not a giant model. Every claim is
embedded as a composed "retrieval card" whose lead text is the agent-authored
``description`` (written for search — symptoms, synonyms, scope), followed by the
structured signal (subject • predicate • object • scope • fact). This module is
the **single** card builder, shared by the write path (embed-on-write) and any
eval read path, so the text indexed and the text searched are composed the same
way.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

_SEP = " • "


def humanize_entity_key(key: str | None) -> str:
    """Turn ``service:payments-api`` into ``payments api`` for embedding text."""
    if not key:
        return ""
    body = key.split(":", 1)[1] if ":" in key else key
    return body.replace("-", " ").replace("_", " ").replace("/", " ").strip()


def build_retrieval_card(
    *,
    description: str | None = None,
    fact: str | None = None,
    subject_key: str | None = None,
    predicate: str | None = None,
    object_key: str | None = None,
    object_value: str | None = None,
    scope: Mapping[str, Any] | None = None,
    extra_terms: Sequence[str] = (),
) -> str:
    """Compose the canonical retrieval card for a claim.

    Order matters: the agent-authored ``description`` leads (it carries the
    search-shaped signal), then the structured fields fill in entities, the
    relation, and scope so a query that names any of them still matches.
    """
    parts: list[str] = []

    def add(value: str | None) -> None:
        if value and value.strip() and value.strip() not in parts:
            parts.append(value.strip())

    add(description)
    if fact and fact != description:
        add(fact)
    add(humanize_entity_key(subject_key))
    if predicate:
        add(predicate.lower().replace("_", " "))
    add(humanize_entity_key(object_key))
    add(object_value)
    if scope:
        for value in scope.values():
            if isinstance(value, str):
                add(value)
    for term in extra_terms:
        add(term)
    return _SEP.join(parts)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [0, 1] (clamped). Returns 0 for empty/degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    sim = dot / (math.sqrt(na) * math.sqrt(nb))
    # Cosine of non-negative-ish feature vectors lands in [-1, 1]; clamp to
    # [0, 1] so it composes with the other [0, 1] ranking factors.
    return max(0.0, min(1.0, sim))


__all__ = ["build_retrieval_card", "cosine_similarity", "humanize_entity_key"]
