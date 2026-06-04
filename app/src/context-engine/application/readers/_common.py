"""Shared scaffolding for the P9 use-case readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

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


__all__ = [
    "ReadRequest",
    "ReadResponse",
    "coverage_status_from_count",
    "make_task_context",
    "rank_candidates",
]
