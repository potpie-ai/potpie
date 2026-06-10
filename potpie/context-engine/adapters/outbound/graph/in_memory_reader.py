"""In-memory :class:`ClaimQueryPort` for tests + offline benches.

Production uses the Neo4j adapter against the canonical edge store. This
implementation is *not* a substitute for that — it covers the filter semantics
defined on :class:`ClaimQueryFilter` so reader tests can drive every code path
without a live graph.

Semantic match (R1): when an :class:`EmbedderPort` is wired, ``fact_query`` runs
a real cosine-similarity ranking over the persisted ``fact_embedding`` (embedding
the retrieval card on the fly for rows that lack one). With no embedder it falls
back to a **labeled** Jaccard token-overlap (``match_mode == "lexical"``), so an
empty result is debuggable rather than a silent stub.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.ports.embedder import EmbedderPort
from domain.retrieval_card import build_retrieval_card, cosine_similarity


@dataclass(slots=True)
class InMemoryClaimQueryStore:
    rows: list[ClaimRow] = field(default_factory=list)
    entity_label_index: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict
    )
    entity_property_index: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    embedder: EmbedderPort | None = None

    @property
    def match_mode(self) -> str:
        """Active semantic-match mode: ``vector`` (embedder) or ``lexical``."""
        return "vector" if self.embedder is not None else "lexical"

    # ------------------------------------------------------------------
    # Loading helpers (used by tests + benches to populate the store)
    # ------------------------------------------------------------------
    def add(self, row: ClaimRow) -> None:
        self.rows.append(row)

    def add_many(self, rows: Iterable[ClaimRow]) -> None:
        self.rows.extend(rows)

    def set_entity_label(
        self, *, pot_id: str, entity_key: str, labels: Iterable[str]
    ) -> None:
        self.entity_label_index[(pot_id, entity_key)] = tuple(labels)

    def set_entity_properties(
        self, *, pot_id: str, entity_key: str, properties: Mapping[str, Any]
    ) -> None:
        """Merge entity properties (name / description / …) for richer reads."""
        if not properties:
            return
        bucket = self.entity_property_index.setdefault((pot_id, entity_key), {})
        bucket.update({k: v for k, v in properties.items()})

    def entity_properties(self, *, pot_id: str, entity_key: str) -> dict[str, Any]:
        return dict(self.entity_property_index.get((pot_id, entity_key), {}))

    # ------------------------------------------------------------------
    # ClaimQueryPort
    # ------------------------------------------------------------------
    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]:
        candidates = [row for row in self.rows if row.pot_id == filter_.pot_id]
        candidates = [row for row in candidates if _matches_filter(row, filter_, self)]

        if filter_.fact_query:
            candidates = self._semantic_rank(candidates, filter_.fact_query)

        if filter_.limit is not None and filter_.limit >= 0:
            candidates = candidates[: filter_.limit]
        return candidates

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        for key in entity_keys:
            labels = self.entity_label_index.get((pot_id, key))
            if labels:
                out[key] = labels
        return out

    # ------------------------------------------------------------------
    # Semantic ranking
    # ------------------------------------------------------------------
    def _semantic_rank(self, candidates: list[ClaimRow], query: str) -> list[ClaimRow]:
        if self.embedder is not None:
            qvec = self.embedder.embed(query)
            scored = [
                (row, cosine_similarity(qvec, self._row_vector(row)))
                for row in candidates
            ]
        else:
            scored = [(row, _embedding_score(row, query)) for row in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [_stamp_similarity(row, score) for row, score in scored]

    def _row_vector(self, row: ClaimRow) -> tuple[float, ...]:
        if row.fact_embedding:
            return row.fact_embedding
        # No persisted embedding (old data / hand-built test row): embed the
        # card on the fly so the row still participates in vector ranking.
        assert self.embedder is not None
        return self.embedder.embed(card_for_row(row))


def card_for_row(row: ClaimRow) -> str:
    """Build the retrieval card for a stored claim row (read-side / on-the-fly embed)."""
    props = row.properties or {}
    return build_retrieval_card(
        description=row.description or props.get("description"),
        fact=row.fact,
        subject_key=row.subject_key,
        predicate=row.predicate,
        object_key=row.object_key,
        scope=props.get("code_scope") if isinstance(props.get("code_scope"), Mapping) else None,
    )


def _stamp_similarity(row: ClaimRow, score: float) -> ClaimRow:
    props = dict(row.properties)
    props["semantic_similarity"] = score
    return dataclasses.replace(row, properties=props)


def _matches_filter(
    row: ClaimRow, filter_: ClaimQueryFilter, store: InMemoryClaimQueryStore
) -> bool:
    if filter_.predicate_in and row.predicate not in filter_.predicate_in:
        return False
    if filter_.subject_key_in and row.subject_key not in filter_.subject_key_in:
        return False
    if filter_.object_key_in and row.object_key not in filter_.object_key_in:
        return False
    if filter_.source_system_in and row.source_system not in filter_.source_system_in:
        return False
    if not filter_.include_invalidated and row.invalid_at is not None:
        return False
    if filter_.valid_at_after and (
        row.valid_at is None or row.valid_at < filter_.valid_at_after
    ):
        return False
    if filter_.valid_at_before and (
        row.valid_at is not None and row.valid_at > filter_.valid_at_before
    ):
        return False
    if filter_.as_of and row.valid_at is not None and row.valid_at > filter_.as_of:
        return False
    if filter_.subject_label:
        labels = store.entity_label_index.get((row.pot_id, row.subject_key), ())
        if filter_.subject_label not in labels:
            return False
    if filter_.object_label:
        labels = store.entity_label_index.get((row.pot_id, row.object_key), ())
        if filter_.object_label not in labels:
            return False
    return True


def _embedding_score(row: ClaimRow, query: str) -> float:
    """Labeled lexical fallback: token-overlap Jaccard over the retrieval card.

    Used only when no :class:`EmbedderPort` is wired (``match_mode='lexical'``).
    Scores the whole retrieval card (not just ``fact``) so the structured signal
    still contributes, but it remains lexical — paraphrases score low. Named, not
    silent: callers can see ``match_mode`` in status.
    """
    text = card_for_row(row) or (row.fact or "")
    a = set(query.lower().split())
    b = set(text.lower().split())
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    base = len(intersection) / len(union)
    if intersection:
        base = math.sqrt(base)
    return max(0.0, min(1.0, base))


__all__ = ["InMemoryClaimQueryStore", "card_for_row"]
