"""Shared scaffolding for the P9 use-case readers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from domain.ports.claim_query import ClaimRow
from domain.ranking import Candidate, RankedItem, RankingService, TaskContext


@dataclass(frozen=True, slots=True)
class ReadRequest:
    """Reader input — keeps callers from depending on the global query model."""

    pot_id: str
    scope: Mapping[str, Any] = field(default_factory=dict)
    query: str | None = None
    intent: str | None = None
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    max_items: int = 12
    freshness_preference: str = "balanced"
    include_invalidated: bool = False
    source_refs: tuple[str, ...] = ()
    # Traverse-axis controls (Query Surface). Only the neighborhood reader uses
    # these; other readers ignore them.
    depth: int | None = None
    direction: str | None = None  # "out" | "in" | "both"


@dataclass(frozen=True, slots=True)
class ReadResponse:
    """Reader output — a ranked list plus per-reader meta."""

    family: str
    items: tuple[RankedItem, ...]
    coverage_status: str
    meta: Mapping[str, Any] = field(default_factory=dict)


def make_task_context(req: ReadRequest) -> TaskContext:
    return TaskContext(
        pot_id=req.pot_id,
        scope=req.scope,
        intent=req.intent,
        freshness_preference=req.freshness_preference,
        now=req.as_of,
    )


def coverage_status_from_count(*, found: int, requested: int) -> str:
    """Translate hit-counts into the F5 coverage label.

    - ``empty``: nothing returned at all.
    - ``sparse``: > 0 but < 50% of the request.
    - ``partial``: ≥ 50% and < 100% of the request.
    - ``complete``: ≥ requested.
    """
    if found <= 0:
        return "empty"
    if requested <= 0:
        return "complete"
    ratio = found / requested
    if ratio >= 1.0:
        return "complete"
    if ratio >= 0.5:
        return "partial"
    return "sparse"


def rank_candidates(
    *, service: RankingService, candidates: Iterable[Candidate], req: ReadRequest
) -> list[RankedItem]:
    ctx = make_task_context(req)
    ranked = service.rank(candidates, ctx)
    if req.max_items > 0:
        ranked = ranked[: req.max_items]
    return ranked


def claim_candidate_key(row: ClaimRow) -> str:
    return row.claim_key or f"{row.predicate}:{row.subject_key}:{row.object_key}"


def dedupe_claim_rows(rows: Iterable[ClaimRow]) -> list[ClaimRow]:
    """Preserve first occurrence of duplicate backend claim rows."""
    seen: set[tuple[Any, ...]] = set()
    out: list[ClaimRow] = []
    for row in rows:
        key = _claim_row_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _claim_row_dedupe_key(row: ClaimRow) -> tuple[Any, ...]:
    if row.claim_key:
        return ("claim", row.claim_key)
    source_refs = row.source_refs
    if not source_refs and row.source_ref:
        source_refs = (row.source_ref,)
    return (
        "triple",
        row.predicate.upper(),
        row.subject_key,
        row.object_key,
        tuple(sorted(source_refs)),
    )


def claim_corroboration(row: ClaimRow) -> int:
    count = row.properties.get("corroboration_count")
    if isinstance(count, int) and count > 0:
        return count
    return 1


def claim_environment(row: ClaimRow) -> str | None:
    env = row.environment
    if isinstance(env, str) and env.strip():
        return env.strip().lower()
    return None


def claim_payload(
    row: ClaimRow,
    *,
    environment: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "description": row.description,
        "fact": row.fact,
        "environment": (
            environment if environment is not None else claim_environment(row)
        ),
        "source_refs": list(
            row.source_refs or ((row.source_ref,) if row.source_ref else ())
        ),
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "valid_until": row.valid_until.isoformat() if row.valid_until else None,
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
        "evidence_strength": row.evidence_strength,
    }
    if extra:
        payload.update(extra)
    return payload


def service_anchor_keys(
    scope: Mapping[str, Any],
    *,
    include_anchor_entity_key: bool = False,
) -> list[str]:
    return scoped_entity_keys(
        scope,
        prefixes=("service",),
        include_anchor_entity_key=include_anchor_entity_key,
    )


def scoped_entity_keys(
    scope: Mapping[str, Any],
    *,
    prefixes: Iterable[str],
    include_anchor_entity_key: bool = False,
) -> list[str]:
    keys: list[str] = []
    for prefix in prefixes:
        values = scope.get(f"{prefix}s") or scope.get(prefix)
        if isinstance(values, str):
            values = (values,)
        elif not isinstance(values, Iterable) or isinstance(values, Mapping):
            values = ()
        for value in values:
            if not (isinstance(value, str) and value.strip()):
                continue
            key = value.strip()
            keys.append(key if ":" in key else f"{prefix}:{key.lower()}")
    if include_anchor_entity_key and not keys:
        anchor = scope.get("anchor_entity_key")
        if isinstance(anchor, str) and anchor.strip():
            keys.append(anchor.strip())
    return list(dict.fromkeys(keys))


def row_in_anchor_set(row: ClaimRow, anchor_keys: Iterable[str]) -> bool:
    anchors = set(anchor_keys)
    return row.subject_key in anchors or row.object_key in anchors


__all__ = [
    "ReadRequest",
    "ReadResponse",
    "claim_candidate_key",
    "claim_corroboration",
    "claim_environment",
    "claim_payload",
    "coverage_status_from_count",
    "dedupe_claim_rows",
    "make_task_context",
    "rank_candidates",
    "row_in_anchor_set",
    "scoped_entity_keys",
    "service_anchor_keys",
]
