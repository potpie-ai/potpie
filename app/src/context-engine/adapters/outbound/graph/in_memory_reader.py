"""In-memory :class:`ClaimQueryPort` for tests + offline benches.

Production uses the Neo4j adapter against the canonical edge store.
This implementation is *not* a substitute for that — it covers the
filter semantics defined on :class:`ClaimQueryFilter` so reader tests
can drive every code path without a live graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from domain.ports.claim_query import ClaimQueryFilter, ClaimRow


@dataclass(slots=True)
class InMemoryClaimQueryStore:
    rows: list[ClaimRow] = field(default_factory=list)
    entity_label_index: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict
    )

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

    # ------------------------------------------------------------------
    # ClaimQueryPort
    # ------------------------------------------------------------------
    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]:
        candidates = [row for row in self.rows if row.pot_id == filter_.pot_id]
        candidates = [row for row in candidates if _matches_filter(row, filter_, self)]

        if filter_.fact_query:
            scored = [(row, _embedding_score(row, filter_.fact_query)) for row in candidates]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            stamped: list[ClaimRow] = []
            for row, score in scored:
                props = dict(row.properties)
                props["semantic_similarity"] = score
                stamped.append(
                    ClaimRow(
                        pot_id=row.pot_id,
                        predicate=row.predicate,
                        subject_key=row.subject_key,
                        object_key=row.object_key,
                        valid_at=row.valid_at,
                        invalid_at=row.invalid_at,
                        evidence_strength=row.evidence_strength,
                        source_system=row.source_system,
                        source_ref=row.source_ref,
                        fact=row.fact,
                        properties=props,
                        fact_embedding=row.fact_embedding,
                    )
                )
            candidates = stamped

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
    """Stand-in similarity: token-overlap Jaccard on lowercased fact + query."""
    if not row.fact:
        return 0.0
    a = set(query.lower().split())
    b = set(row.fact.lower().split())
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    base = len(intersection) / len(union)
    # Slight boost so close-but-not-identical phrases still score above 0
    if intersection:
        base = math.sqrt(base)
    return max(0.0, min(1.0, base))


__all__ = ["InMemoryClaimQueryStore"]
