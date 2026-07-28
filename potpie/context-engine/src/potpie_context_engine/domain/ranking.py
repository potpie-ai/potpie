"""Uniform ranker (rebuild plan P7).

Every retrieval scores candidate claims/entities by
``strength × recency × scope_overlap × corroboration × semantic_similarity``,
with a coverage-quality downweight that flows from F5. The same ranker
serves every reader — readers extract the per-factor inputs (since the
factors are use-case specific), the ranker applies a uniform formula
over them.

This module is intentionally I/O-free: callers build :class:`Candidate`
objects from graph reads, hand them to :class:`RankingService`, and get
back :class:`RankedItem` objects with the score breakdown attached.
Breaking down the score lets readers explain rankings to the agent
(plan exit criterion: "higher-strength wins, every time, deterministically").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence


# Default weights. Semantic similarity is the **primary recall signal** (R1/R3):
# it carries the highest weight because, with embed-on-write + ANN, it is the
# factor that actually separates a paraphrased match from noise. The rest
# re-rank. ``scope_overlap`` is high too (UC1 hinges on it).
#
# Combination rule (R3 — the single, documented rule): a **weighted arithmetic
# mean** of the per-factor scores (each in [0, 1]). This deliberately replaces
# the old weighted *geometric* mean, whose ``1e-6`` floor let a single zero (a
# low lexical-overlap score) veto an otherwise strong, recent, scope-matched
# claim. Under a weighted sum a weak factor only re-ranks; it can never collapse
# a candidate. Hard filters (pot / predicate / validity window / scope
# membership) belong in the query, not here — see the readers.
_DEFAULT_WEIGHTS: Mapping[str, float] = {
    "semantic_similarity": 1.3,
    "strength": 1.2,
    "scope_overlap": 1.1,
    "recency": 1.0,
    "corroboration": 0.8,
    "coverage_quality": 0.5,
}


_STRENGTH_TO_SCORE: dict[str, float] = {
    "deterministic": 1.00,
    "attested": 0.80,
    "stated": 0.60,
    "inferred": 0.45,
    "speculative": 0.20,
}


_DEFAULT_RECENCY_HALF_LIFE = timedelta(days=30)


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Caller-supplied task framing the ranker compares candidates against."""

    pot_id: str
    scope: Mapping[str, Any] = field(default_factory=dict)
    intent: str | None = None
    freshness_preference: str = "balanced"  # 'fresh' | 'balanced' | 'historical'
    now: datetime | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    """One claim/entity the ranker scores.

    Each per-factor input is optional. When absent, the ranker uses a
    neutral midpoint (0.5) so a missing signal neither helps nor hurts.
    Readers are expected to populate the factors they can compute
    deterministically and leave the rest unset.
    """

    candidate_key: str
    payload: Mapping[str, Any]  # whatever the reader wants to surface
    strength: str | None = None
    valid_at: datetime | None = None
    corroboration_count: int = 1
    scope_overlap: float | None = None  # [0, 1]
    semantic_similarity: float | None = None  # [0, 1]
    coverage_status: str | None = None  # 'complete' | 'partial' | 'sparse' | 'empty'
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedItem:
    """A scored candidate with its per-factor breakdown attached."""

    candidate: Candidate
    score: float
    breakdown: Mapping[str, float]


@dataclass(slots=True)
class RankingService:
    """Stateless ranker. Inject custom weights via the constructor for tuning."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    strength_scores: Mapping[str, float] = field(
        default_factory=lambda: dict(_STRENGTH_TO_SCORE)
    )
    recency_half_life: timedelta = _DEFAULT_RECENCY_HALF_LIFE

    def rank(
        self,
        candidates: Iterable[Candidate],
        context: TaskContext,
    ) -> list[RankedItem]:
        """Return ``candidates`` ordered by descending score.

        Each candidate's score is recorded in the returned item's
        ``breakdown`` so readers can surface "why this ranked high" if
        the agent asks.
        """
        now = context.now or datetime.now(tz=timezone.utc)
        ranked: list[RankedItem] = []
        for cand in candidates:
            breakdown = self._score_one(cand, now=now, context=context)
            score = self._combine(breakdown)
            ranked.append(RankedItem(candidate=cand, score=score, breakdown=breakdown))
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_one(
        self,
        cand: Candidate,
        *,
        now: datetime,
        context: TaskContext,
    ) -> dict[str, float]:
        return {
            "strength": _strength_score(cand.strength, self.strength_scores),
            "recency": _recency_score(
                cand.valid_at,
                now=now,
                half_life=self.recency_half_life,
                preference=context.freshness_preference,
            ),
            "scope_overlap": _clamp(cand.scope_overlap, default=0.5),
            "corroboration": _corroboration_score(cand.corroboration_count),
            "semantic_similarity": _clamp(cand.semantic_similarity, default=0.5),
            "coverage_quality": _coverage_quality_score(cand.coverage_status),
        }

    def _combine(self, breakdown: Mapping[str, float]) -> float:
        """Weighted arithmetic mean of the per-factor scores (R3).

        No single soft signal can veto a candidate: a zero factor contributes
        zero to the sum rather than collapsing the product. Stays in [0, 1] and
        is stable across reader output sizes.
        """
        weighted = 0.0
        weight_sum = 0.0
        for factor, value in breakdown.items():
            weight = self.weights.get(factor, 0.0)
            if weight <= 0:
                continue
            weighted += weight * max(0.0, min(1.0, value))
            weight_sum += weight
        if weight_sum == 0:
            return 0.0
        return weighted / weight_sum


# ---------------------------------------------------------------------------
# Per-factor scoring (module-private helpers)
# ---------------------------------------------------------------------------


def _clamp(value: float | None, *, default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def _strength_score(strength: str | None, mapping: Mapping[str, float]) -> float:
    if strength is None:
        return 0.5
    return mapping.get(strength, 0.5)


def _recency_score(
    valid_at: datetime | None,
    *,
    now: datetime,
    half_life: timedelta,
    preference: str = "balanced",
) -> float:
    """Exponential decay; freshness preference shifts the half-life."""
    if valid_at is None:
        return 0.5
    if valid_at.tzinfo is None:
        valid_at = valid_at.replace(tzinfo=timezone.utc)
    age = max(now - valid_at, timedelta(0))

    effective_half_life = half_life
    if preference == "fresh":
        effective_half_life = max(half_life / 2, timedelta(days=1))
    elif preference == "historical":
        effective_half_life = half_life * 4

    half_life_seconds = effective_half_life.total_seconds()
    if half_life_seconds <= 0:
        return 0.5
    decay = 0.5 ** (age.total_seconds() / half_life_seconds)
    return max(0.0, min(1.0, decay))


def _corroboration_score(count: int) -> float:
    """Diminishing returns: one source → 0.5, two → 0.71, three → 0.79, …"""
    if count is None or count <= 0:
        return 0.0
    if count == 1:
        return 0.5
    return min(1.0, 1 - (1 / (count + 1)))


def _coverage_quality_score(status: str | None) -> float:
    if status is None:
        return 0.75
    return {
        "complete": 1.0,
        "partial": 0.65,
        "sparse": 0.4,
        "empty": 0.1,
    }.get(status, 0.5)


# ---------------------------------------------------------------------------
# Public convenience helpers — readers use these to build Candidates from
# the canonical edge property bag without re-implementing the mapping.
# ---------------------------------------------------------------------------


def candidate_from_edge_record(
    *,
    candidate_key: str,
    edge_properties: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
    scope_overlap: float | None = None,
    semantic_similarity: float | None = None,
    coverage_status: str | None = None,
    corroboration_count: int | None = None,
) -> Candidate:
    """Translate one Position-B edge property bag into a :class:`Candidate`.

    The reader pulls the edge record from a Cypher query and hands the
    full property dict; this helper extracts the canonical fields
    (``evidence_strength``, ``valid_at``) without the reader needing to
    know about the column-name conventions.
    """
    strength = edge_properties.get("evidence_strength")
    if not isinstance(strength, str):
        strength = None

    valid_at_raw = edge_properties.get("valid_at")
    if isinstance(valid_at_raw, datetime):
        valid_at = valid_at_raw
    elif isinstance(valid_at_raw, str):
        try:
            valid_at = datetime.fromisoformat(valid_at_raw)
        except ValueError:
            valid_at = None
    else:
        valid_at = None

    return Candidate(
        candidate_key=candidate_key,
        payload=dict(payload or {}),
        strength=strength,
        valid_at=valid_at,
        corroboration_count=int(corroboration_count) if corroboration_count else 1,
        scope_overlap=scope_overlap,
        semantic_similarity=semantic_similarity,
        coverage_status=coverage_status,
    )


def truncate(items: Sequence[RankedItem], *, max_items: int) -> list[RankedItem]:
    """Convenience: keep the top ``max_items`` after ranking."""
    if max_items <= 0:
        return []
    return list(items[:max_items])


__all__ = [
    "Candidate",
    "RankedItem",
    "RankingService",
    "TaskContext",
    "candidate_from_edge_record",
    "truncate",
]
