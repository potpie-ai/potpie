"""Shared canonical claim-query helpers for Cypher graph adapters."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from datetime import datetime, timezone

from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter, ClaimRow

from potpie_context_engine.core.claim_codec import (
    CONTRACT_EDGE_KEYS,
    RESERVED_EDGE_KEYS,
    _as_utc,
    iso,
    parse_dt,
    row_from_record,
    vector_property,
)


def claim_is_applicable(row: ClaimRow, *, as_of: datetime | None = None) -> bool:
    """Return whether a claim applies in the half-open valid-time interval."""
    point = _as_utc(as_of or datetime.now(timezone.utc))
    start = _as_utc(row.valid_at) if row.valid_at is not None else None
    valid_until = _as_utc(row.valid_until) if row.valid_until is not None else None
    invalid_at = _as_utc(row.invalid_at) if row.invalid_at is not None else None
    if start is not None and start > point:
        return False
    if valid_until is not None and point >= valid_until:
        return False
    if invalid_at is not None and point >= invalid_at:
        return False
    return True


def claim_matches_time_filter(
    row: ClaimRow, filter_: ClaimQueryFilter, *, query_time: datetime
) -> bool:
    """Compare instants rather than the lexical order of stored ISO offsets."""
    if not filter_.include_invalidated and not claim_is_applicable(
        row, as_of=query_time
    ):
        return False
    if filter_.valid_at_after is not None and (
        row.valid_at is None or _as_utc(row.valid_at) < _as_utc(filter_.valid_at_after)
    ):
        return False
    return (
        filter_.valid_at_before is None
        or row.valid_at is None
        or _as_utc(row.valid_at) <= _as_utc(filter_.valid_at_before)
    )


def claim_matches_post_filter(
    row: ClaimRow, filter_: ClaimQueryFilter, *, query_time: datetime
) -> bool:
    """Hydrated predicates that must run before ranking and the result limit."""
    if not claim_matches_time_filter(row, filter_, query_time=query_time):
        return False
    for values, actual in (
        (filter_.environment_in, row.environment),
        (filter_.truth_in, row.truth),
    ):
        if values and (actual or "").lower() not in {value.lower() for value in values}:
            return False
    if filter_.exact_text_in or filter_.exact_text_pattern:
        import re

        haystack = " ".join(
            str(value)
            for value in (
                row.subject_key,
                row.object_key,
                row.claim_key,
                row.fact,
                row.description,
                row.source_ref,
                row.source_refs,
                row.properties,
            )
            if value is not None
        ).lower()
        if filter_.exact_text_in and not any(
            needle.lower() in haystack for needle in filter_.exact_text_in
        ):
            return False
        if (
            filter_.exact_text_pattern
            and re.fullmatch(filter_.exact_text_pattern, haystack) is None
        ):
            return False
    return True


def embedding_score(fact: str | None, query: str) -> float:
    """Token-overlap stand-in for native vector similarity."""
    if not fact:
        return 0.0
    a = set(query.lower().split())
    b = set(fact.lower().split())
    if not a or not b:
        return 0.0
    base = len(a & b) / len(a | b)
    if a & b:
        base = math.sqrt(base)
    return max(0.0, min(1.0, base))


def stamp_similarity(rows: Iterable[ClaimRow], query: str) -> list[ClaimRow]:
    """Sort rows by lexical similarity and stamp ``semantic_similarity``."""
    scored = sorted(
        ((embedding_score(row.fact, query), row) for row in rows),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return stamp_scored_rows(scored)


def stamp_scored_rows(scored: Iterable[tuple[float, ClaimRow]]) -> list[ClaimRow]:
    """Stamp precomputed scores onto rows, preserving score order."""
    out: list[ClaimRow] = []
    for score, row in scored:
        props = dict(row.properties)
        props["semantic_similarity"] = float(score)
        out.append(dataclasses.replace(row, properties=props))
    return out


# Concatenating with [] preserves native arrays and wraps legacy scalar
# source_refs as one element before any() (coalesce alone does not coerce types).
# Keep the vector query predicates consistent with this scan predicate.
#
# ``fact_embedding: NULL`` overrides the ``.*`` projection: embeddings are a
# write/index-side concern, and shipping a full vector per row dominates the
# reply size (and, on redis backends, RESP parse time) of every claim scan.
FIND_CLAIMS_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO]->(b:Entity {group_id: $gid})
WHERE ($include_retired OR (coalesce(r.retired,false)=false AND coalesce(a.retired,false)=false AND coalesce(b.retired,false)=false))
  AND ($preds IS NULL OR r.name IN $preds)
  AND ($subjects IS NULL OR r.subject_key IN $subjects)
  AND ($objects IS NULL OR r.object_key IN $objects)
  AND ($claim_keys IS NULL OR r.claim_key IN $claim_keys)
  AND ($record_ids IS NULL OR coalesce(r.record_id, r.uuid) IN $record_ids)
  AND ($subgraphs IS NULL OR r.subgraph IN $subgraphs)
  AND ($excluded_subgraphs IS NULL OR NOT (r.subgraph IN $excluded_subgraphs))
  AND ($mutation_ids IS NULL OR r.mutation_id IN $mutation_ids)
  AND ($source_refs IS NULL OR r.source_ref IN $source_refs OR any(ref IN [] + coalesce(r.source_refs, []) WHERE ref IN $source_refs))
  AND ($sources IS NULL OR r.source_system IN $sources)
  AND ($endpoint_label IS NULL OR $endpoint_label IN labels(a) OR $endpoint_label IN labels(b))
  AND ($subject_label IS NULL OR $subject_label IN labels(a))
  AND ($object_label IS NULL OR $object_label IN labels(b))
RETURN r{.*, fact_embedding: NULL} AS props
"""


# ANN returns top-k by distance, then WHERE filters predicates/subgraphs. A
# selective filter (predicate allowlist or subgraph exclusion) needs a larger
# candidate pool so a dense neighborhood of out-of-family claims cannot empty
# the post-filter result (resources P9 — section corpus crowding).
_VECTOR_K_BASE_MULTIPLIER = 5
_VECTOR_K_BASE_FLOOR = 50
_VECTOR_K_SELECTIVE_MULTIPLIER = 25
_VECTOR_K_SELECTIVE_FLOOR = 250


def vector_candidate_k(limit: int, *, selective: bool) -> int:
    """How many ANN neighbors to pull before applying claim filters."""
    bound = max(1, int(limit))
    if selective:
        return max(bound * _VECTOR_K_SELECTIVE_MULTIPLIER, _VECTOR_K_SELECTIVE_FLOOR)
    return max(bound * _VECTOR_K_BASE_MULTIPLIER, _VECTOR_K_BASE_FLOOR)


def vector_filter_is_selective(filter_: ClaimQueryFilter) -> bool:
    return bool(filter_.predicate_in or filter_.subgraph_not_in)


ENTITY_LABELS_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key IN $keys AND coalesce(e.retired,false)=false
RETURN e.entity_key AS key, labels(e) AS labels
"""


__all__ = [
    "CONTRACT_EDGE_KEYS",
    "ENTITY_LABELS_CYPHER",
    "FIND_CLAIMS_CYPHER",
    "RESERVED_EDGE_KEYS",
    "embedding_score",
    "iso",
    "parse_dt",
    "row_from_record",
    "stamp_scored_rows",
    "stamp_similarity",
    "vector_candidate_k",
    "vector_filter_is_selective",
    "vector_property",
]
