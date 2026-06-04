"""Belief derivation + coverage-gap-aware confidence.

Rebuild plan P2: the substrate already supports per-claim writes
(Position B in P0); this module computes the *belief* over a set of
live claims for a given ``(subject, predicate)`` — what the envelope
returns to the agent as ``confidence``.

Inputs the formula consumes:

- ``evidence_strength`` — ordered ``deterministic > attested > inferred
  > hypothesized``. Scanner-derived claims carry ``deterministic``;
  LLM-extracted from PR body carry ``inferred``; agent-recorded carry
  ``attested``.
- ``valid_at`` recency vs. now and the predicate-family TTL.
- Per-source authority weight — k8s-scanner outranks slack-message.
- Corroboration count across distinct ``(source_system, source_ref)``.
- Verification claims attached to the target (``VERIFIED`` predicate).
- **Coverage-gap signal (F5)** — when the planner queried for N
  evidence families and ≥1 came back empty, the envelope's *top-level*
  confidence is capped at ``medium`` even if individual claims are
  high-strength. Single most-load-bearing change vs. the substrate POC.

Output shape:

- ``per_fact_label`` ∈ ``{"high","medium","low","unknown"}`` for each
  belief candidate.
- ``envelope_confidence`` is the cross-leg cap'd label the envelope
  surfaces; it is *never* higher than the highest per-fact label and is
  capped down by coverage gaps.

The formula is intentionally legible (weighted product over labelled
factors) and deterministic; per-source weights and decay TTLs live in
constants so bench data drives tuning without code edits in the hot
path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Iterable, Mapping


# ---------------------------------------------------------------------------
# Labels + helpers
# ---------------------------------------------------------------------------


class ConfidenceLabel(str):
    """Marker-pattern label values; using str subclass for JSON-friendliness."""


HIGH = "high"
MEDIUM = "medium"
LOW = "low"
UNKNOWN = "unknown"

# Numeric rank for cross-comparison; never user-visible.
_LABEL_RANK: Mapping[str, int] = {UNKNOWN: 0, LOW: 1, MEDIUM: 2, HIGH: 3}


def confidence_min(a: str, b: str) -> str:
    """Return the *lower* of two confidence labels (cap-down semantics)."""
    return a if _LABEL_RANK.get(a, 0) <= _LABEL_RANK.get(b, 0) else b


def confidence_max(labels: Iterable[str]) -> str:
    """Return the *highest* of a label sequence; ``UNKNOWN`` when empty."""
    best = UNKNOWN
    best_rank = _LABEL_RANK[UNKNOWN]
    for label in labels:
        rank = _LABEL_RANK.get(label, 0)
        if rank > best_rank:
            best = label
            best_rank = rank
    return best


class EvidenceStrength(IntEnum):
    """Ordered evidence-strength rank used in the belief formula."""

    HYPOTHESIZED = 1
    INFERRED = 2
    ATTESTED = 3
    DETERMINISTIC = 4


_STRENGTH_LOOKUP: Mapping[str, EvidenceStrength] = {
    "hypothesized": EvidenceStrength.HYPOTHESIZED,
    "inferred": EvidenceStrength.INFERRED,
    "attested": EvidenceStrength.ATTESTED,
    "deterministic": EvidenceStrength.DETERMINISTIC,
}


def evidence_strength_value(name: str | None) -> int:
    """Map a strength name to its rank; unknowns default to ``HYPOTHESIZED``."""
    if not name:
        return EvidenceStrength.HYPOTHESIZED.value
    return _STRENGTH_LOOKUP.get(name.lower(), EvidenceStrength.HYPOTHESIZED).value


# ---------------------------------------------------------------------------
# Source authority weights — boundary configuration (tune from bench data).
# Higher = more trusted; 1.0 = neutral. Keep the table small + readable.
# ---------------------------------------------------------------------------


_SOURCE_AUTHORITY: Mapping[str, float] = {
    # High-trust deterministic sources (scanners reading config in-place)
    "k8s-scanner": 1.2,
    "kubernetes-manifest-scanner": 1.2,
    "codeowners-scanner": 1.2,
    "openapi-spec-scanner": 1.15,
    "dependency-manifest-scanner": 1.15,
    # Agent-recorded structured payloads (P6) — high trust per record_type
    "agent-record": 1.1,
    "agent-verification": 1.1,
    # Authoritative event-driven systems
    "github": 1.0,
    "linear": 1.0,
    # Ambient / noisy sources — high recall, low signal density
    "slack-message": 0.7,
    "pr-body-llm": 0.7,
    "doc-extract-llm": 0.7,
}

DEFAULT_SOURCE_AUTHORITY = 1.0


def source_authority(source_system: str | None) -> float:
    """Per-source authority multiplier; unknown sources default to ``1.0``."""
    if not source_system:
        return DEFAULT_SOURCE_AUTHORITY
    return _SOURCE_AUTHORITY.get(source_system.lower(), DEFAULT_SOURCE_AUTHORITY)


# ---------------------------------------------------------------------------
# Family TTLs — predicate-family-aware decay window. Past 2× TTL the
# claim contributes zero unless corroborated; corroboration extends the
# effective TTL by 50% per additional independent source.
# ---------------------------------------------------------------------------


_FAMILY_TTL_HOURS: Mapping[str, float] = {
    # Topology — stable, decays slowly. Recompute frequency matches scanner cadence.
    "DEPENDS_ON": 24 * 30,
    "STORED_IN": 24 * 30,
    "DEPLOYED_TO": 24 * 14,
    "USES": 24 * 30,
    "EXPOSES": 24 * 30,
    "OF_SERVICE": 24 * 90,
    "OWNED_BY": 24 * 90,
    # Activity / process — short-lived, decays fast.
    "PERFORMED": 24 * 7,
    "AUTHORED": 24 * 7,
    "TOUCHED": 24 * 7,
    "MENTIONS": 24 * 14,
    "IN_PERIOD": 24 * 14,
    # Decisions / policies — long-lived but should decay if not re-attested.
    "POLICY_APPLIES_TO": 24 * 365,
    "RESOLVED": 24 * 365,
    "VERIFIED": 24 * 180,
    "SUPERSEDES": 24 * 365 * 5,  # essentially permanent
    "ATTEMPTED_FIX_FAILED": 24 * 90,
    "ALIAS_OF": 24 * 365 * 5,
}

DEFAULT_FAMILY_TTL_HOURS = 24 * 90  # 3 months when the predicate is unknown


def family_ttl_hours(predicate: str | None) -> float:
    """Decay TTL for a predicate family; unknown families default to 90d."""
    if not predicate:
        return DEFAULT_FAMILY_TTL_HOURS
    return _FAMILY_TTL_HOURS.get(predicate, DEFAULT_FAMILY_TTL_HOURS)


def decay_weight(
    *,
    observed_at: datetime | None,
    now: datetime,
    predicate: str | None,
    corroboration_count: int = 1,
) -> float:
    """Linear decay from 1.0 at ``now`` to 0.0 at ``observed_at + 2·TTL``.

    Corroboration extends the effective TTL by 50% per additional
    independent source so well-corroborated facts stay live longer.
    Missing ``observed_at`` is treated as "fresh enough to count" (1.0).
    """
    if observed_at is None:
        return 1.0
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    age = now - observed_at
    if age <= timedelta(0):
        return 1.0
    ttl = family_ttl_hours(predicate)
    extra = max(0, corroboration_count - 1)
    effective_ttl = ttl * (1.0 + 0.5 * extra)
    horizon = timedelta(hours=effective_ttl * 2)
    if age >= horizon:
        return 0.0
    return 1.0 - (age / horizon)


# ---------------------------------------------------------------------------
# Claim + belief value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """One live ``:RELATES_TO`` claim as it arrives from a reader.

    The reader is responsible for filtering to the live set (``invalid_at
    IS NULL`` or ``invalid_at > as_of``) and for resolving the subject /
    predicate / object. The belief deriver only sees the inputs it needs
    to score; this keeps the formula independent of where the claims came
    from (Cypher query, vector search, hybrid merge).
    """

    subject_key: str
    predicate: str
    object_key: str
    source_system: str | None
    source_ref: str | None
    evidence_strength: str
    observed_at: datetime | None
    valid_at: datetime | None = None
    confidence: float | None = None  # rare; usually derived not stored
    verified_count: int = 0  # ``VERIFIED`` claims attached to the target


@dataclass(frozen=True, slots=True)
class BeliefCandidate:
    """One candidate object for a ``(subject, predicate)`` belief.

    ``score`` is the raw weighted number; ``label`` is the coarse user-
    visible value. ``contributing_claims`` is preserved so the envelope
    can render provenance without an extra Cypher round-trip.
    """

    object_key: str
    score: float
    label: str
    corroboration_count: int
    contributing_claims: tuple[ClaimRecord, ...]


@dataclass(frozen=True, slots=True)
class Belief:
    """Aggregated belief for one ``(subject, predicate)``.

    ``winner`` is the highest-scored candidate. ``candidates`` is the
    ranked list (winner first); when there is *equal-recency conflict*
    between candidates ``conflict_open`` is set so the envelope can
    surface a typed quality issue rather than picking arbitrarily.
    """

    subject_key: str
    predicate: str
    winner: BeliefCandidate | None
    candidates: tuple[BeliefCandidate, ...]
    conflict_open: bool = False


# ---------------------------------------------------------------------------
# Formula
# ---------------------------------------------------------------------------


_CORROBORATION_BONUS_PER_EXTRA_SOURCE = 0.5
_CORROBORATION_BONUS_CAP = 1.5
_VERIFICATION_BONUS = 0.75

# Label thresholds — tuned so a single deterministic source ≈ high,
# a single inferred source ≈ low, attested+1 corroboration ≈ medium.
_THRESHOLD_HIGH = 4.0
_THRESHOLD_MEDIUM = 2.5
_THRESHOLD_LOW = 1.0


def score_object(
    claims: list[ClaimRecord],
    *,
    now: datetime,
    verified_count: int = 0,
) -> float:
    """Compute the raw belief score for one candidate object.

    ``score = max(per_claim_score) + corroboration_bonus + verification_bonus``

    Each ``per_claim_score = strength × decay × source_authority`` so a
    fresh deterministic scanner claim from ``k8s-scanner`` reads as
    ``4 × 1.0 × 1.2 = 4.8`` (label = high), while a stale inferred LLM
    claim from a year-old PR body reads as ``2 × 0.0 × 0.7 = 0.0``.
    """
    if not claims:
        return 0.0
    per_claim_scores: list[float] = []
    distinct_sources: set[str] = set()
    for claim in claims:
        strength = float(evidence_strength_value(claim.evidence_strength))
        decay = decay_weight(
            observed_at=claim.observed_at or claim.valid_at,
            now=now,
            predicate=claim.predicate,
            corroboration_count=max(1, len(distinct_sources) or 1),
        )
        authority = source_authority(claim.source_system)
        per_claim_scores.append(strength * decay * authority)
        if claim.source_system:
            distinct_sources.add(claim.source_system.lower())

    base = max(per_claim_scores)
    extras = max(0, len(distinct_sources) - 1)
    bonus = min(
        extras * _CORROBORATION_BONUS_PER_EXTRA_SOURCE,
        _CORROBORATION_BONUS_CAP,
    )
    verification_bonus = (
        _VERIFICATION_BONUS if verified_count > 0 else 0.0
    )
    return base + bonus + verification_bonus


def label_for_score(score: float) -> str:
    """Coarse confidence label for a raw score."""
    if not math.isfinite(score) or score <= 0:
        return UNKNOWN
    if score >= _THRESHOLD_HIGH:
        return HIGH
    if score >= _THRESHOLD_MEDIUM:
        return MEDIUM
    if score >= _THRESHOLD_LOW:
        return LOW
    return UNKNOWN


def derive_belief(
    claims: Iterable[ClaimRecord],
    *,
    subject_key: str,
    predicate: str,
    now: datetime | None = None,
    equal_recency_tolerance: timedelta = timedelta(seconds=30),
) -> Belief:
    """Aggregate claims into one belief for ``(subject, predicate)``.

    Equal-recency conflict: when two candidates have the same most-recent
    ``valid_at`` (within ``equal_recency_tolerance``) AND comparable
    strength, the winner is None and ``conflict_open`` is True. The
    canonical writer's ``apply_family_conflict_detection`` will already
    have stamped a ``QualityIssue`` for these; the belief deriver just
    refuses to pick a side cosmetically.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    # Group by object_key, preserving order for stable output.
    by_object: dict[str, list[ClaimRecord]] = {}
    for claim in claims:
        if claim.subject_key != subject_key or claim.predicate != predicate:
            continue
        by_object.setdefault(claim.object_key, []).append(claim)

    candidates: list[BeliefCandidate] = []
    for object_key, group in by_object.items():
        verified = sum(c.verified_count for c in group)
        score = score_object(group, now=now, verified_count=verified)
        label = label_for_score(score)
        distinct_sources = {
            c.source_system.lower() for c in group if c.source_system
        }
        candidates.append(
            BeliefCandidate(
                object_key=object_key,
                score=score,
                label=label,
                corroboration_count=len(distinct_sources) or len(group),
                contributing_claims=tuple(group),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    if not candidates:
        return Belief(
            subject_key=subject_key,
            predicate=predicate,
            winner=None,
            candidates=(),
        )

    # Equal-recency conflict detection.
    conflict_open = False
    winner: BeliefCandidate | None = candidates[0]
    if len(candidates) >= 2:
        a, b = candidates[0], candidates[1]
        if (
            abs(a.score - b.score) < 0.01
            and _max_recent_observed(a) is not None
            and _max_recent_observed(b) is not None
            and abs(
                (_max_recent_observed(a) - _max_recent_observed(b)).total_seconds()  # type: ignore[operator]
            )
            <= equal_recency_tolerance.total_seconds()
        ):
            conflict_open = True
            winner = None

    return Belief(
        subject_key=subject_key,
        predicate=predicate,
        winner=winner,
        candidates=tuple(candidates),
        conflict_open=conflict_open,
    )


def _max_recent_observed(candidate: BeliefCandidate) -> datetime | None:
    times = [
        c.observed_at or c.valid_at for c in candidate.contributing_claims
    ]
    times = [t for t in times if t is not None]
    if not times:
        return None
    return max(t if t.tzinfo else t.replace(tzinfo=timezone.utc) for t in times)


# ---------------------------------------------------------------------------
# Coverage-gap-aware envelope confidence (F5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Summary of the planner's evidence-family coverage for one resolve.

    ``planned_families`` is the set of evidence families the planner
    expected to be useful. ``returning_families`` is the subset that
    actually returned ≥1 claim. ``empty_families`` is the difference —
    the families that were planned but came back empty. When this set
    is non-empty, the envelope confidence is capped at ``medium`` per
    F5; when *most* families return empty, capped at ``low``.
    """

    planned_families: frozenset[str] = field(default_factory=frozenset)
    returning_families: frozenset[str] = field(default_factory=frozenset)

    @property
    def empty_families(self) -> frozenset[str]:
        return self.planned_families - self.returning_families

    @property
    def status(self) -> str:
        if not self.planned_families:
            # Nothing was planned — treat as full coverage, no cap.
            return "complete"
        ratio = (
            len(self.returning_families) / len(self.planned_families)
        )
        if ratio >= 0.999:
            return "complete"
        if ratio >= 0.5:
            return "partial"
        if ratio > 0:
            return "sparse"
        return "empty"


def coverage_cap(report: CoverageReport) -> str:
    """Maximum envelope confidence allowed under a given coverage report.

    F5: the envelope CANNOT return ``high`` confidence when expected
    families came back empty. Maps coverage status → cap:

    - ``complete`` → cap = ``high`` (no cap)
    - ``partial``  → cap = ``medium``
    - ``sparse``   → cap = ``low``
    - ``empty``    → cap = ``unknown``
    """
    status = report.status
    if status == "complete":
        return HIGH
    if status == "partial":
        return MEDIUM
    if status == "sparse":
        return LOW
    return UNKNOWN


def envelope_confidence(
    per_fact_labels: Iterable[str],
    coverage: CoverageReport,
) -> str:
    """Final envelope confidence = ``min(max(per_fact_labels), coverage_cap)``.

    The envelope's top-level confidence is the lower of (a) the best
    individual belief label and (b) what coverage allows. So a single
    high-confidence claim with most planned families empty downgrades
    to ``low`` or ``unknown`` — the cosmetic-high problem (F5) cannot
    recur.
    """
    best = confidence_max(per_fact_labels)
    return confidence_min(best, coverage_cap(coverage))


__all__ = [
    "HIGH",
    "MEDIUM",
    "LOW",
    "UNKNOWN",
    "Belief",
    "BeliefCandidate",
    "ClaimRecord",
    "CoverageReport",
    "EvidenceStrength",
    "confidence_max",
    "confidence_min",
    "coverage_cap",
    "decay_weight",
    "derive_belief",
    "envelope_confidence",
    "evidence_strength_value",
    "family_ttl_hours",
    "label_for_score",
    "score_object",
    "source_authority",
]
