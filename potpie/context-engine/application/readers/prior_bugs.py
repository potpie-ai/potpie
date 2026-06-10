"""PriorBugsReader (UC4 / P9).

Inputs: symptom signature (free-text or structured) + scope. Logic:
direct native-vector query (via the claim-query port's ``fact_query``)
over ``BugPattern``-related claim edges filtered to live edges and the
agent's scope-or-broader. Rank by ``verification × scope-overlap ×
recency × corroboration`` (the ranker's standard formula, with the
verification signal mapped onto the corroboration factor).

Surfaces both worked fixes (``RESOLVED`` claims) and attempted-failed
fixes (``ATTEMPTED_FIX_FAILED`` claims), the latter labeled so the
agent doesn't repeat them. Hides narrower-scope bugs from broader-
scope queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from application.readers._common import (
    ReadRequest,
    ReadResponse,
    coverage_status_from_count,
    rank_candidates,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ranking import Candidate, RankingService


_BUG_PREDICATES: tuple[str, ...] = (
    "RESOLVED",
    "ATTEMPTED_FIX_FAILED",
    "VERIFIED",
    "REPRODUCES",
)


@dataclass(slots=True)
class PriorBugsReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "prior_bugs"

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = _anchor_keys(req.scope)

        rows = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=req.pot_id,
                predicate_in=_BUG_PREDICATES,
                include_invalidated=req.include_invalidated,
                as_of=req.as_of,
                fact_query=req.query,
                limit=max(req.max_items * 4, 24),
            )
        )

        # Verification-count lookup: a Fix with two VERIFIED claims
        # (different sources) beats a Fix with none. Rather than re-
        # querying, we count VERIFIED rows in the same result set and
        # roll the count into the corroboration_count of the related
        # RESOLVED row.
        verification_counts = _count_verifications(rows)

        candidates: list[Candidate] = []
        for row in rows:
            if row.predicate == "VERIFIED":
                # Verifications fold into their target Fix's score; not
                # surfaced as standalone candidates.
                continue
            overlap = _scope_overlap(row, anchor_keys=anchor_keys)
            if anchor_keys and overlap == 0.0:
                continue
            sim = row.properties.get("semantic_similarity")
            verification_boost = verification_counts.get(row.subject_key, 0)
            candidates.append(
                Candidate(
                    candidate_key=_make_candidate_key(row),
                    payload=_payload_from_row(row, verifications=verification_boost),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    corroboration_count=1 + verification_boost,
                    scope_overlap=overlap if anchor_keys else None,
                    semantic_similarity=float(sim)
                    if isinstance(sim, (int, float))
                    else None,
                )
            )

        ranked = rank_candidates(service=self.ranker, candidates=candidates, req=req)
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={
                "anchor_keys": list(anchor_keys),
                "candidate_pool": len(rows),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anchor_keys(scope: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    services = scope.get("services") or scope.get("service")
    if isinstance(services, str):
        services = [services]
    if isinstance(services, list):
        for s in services:
            if isinstance(s, str) and s.strip():
                keys.append(f"service:{s.strip().lower()}")
    return keys


def _scope_overlap(row: ClaimRow, *, anchor_keys: Iterable[str]) -> float:
    if not anchor_keys:
        return 0.5
    anchors = set(anchor_keys)
    row_scope = row.properties.get("scope_keys")
    if isinstance(row_scope, list):
        hits = sum(1 for k in row_scope if k in anchors)
        if hits:
            return min(1.0, hits / max(len(anchors), 1))
    if row.subject_key in anchors or row.object_key in anchors:
        return 1.0
    return 0.0


def _count_verifications(rows: Iterable[ClaimRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.predicate == "VERIFIED":
            counts[row.subject_key] = counts.get(row.subject_key, 0) + 1
    return counts


def _make_candidate_key(row: ClaimRow) -> str:
    return row.claim_key or f"{row.predicate}:{row.subject_key}:{row.object_key}"


def _payload_from_row(row: ClaimRow, *, verifications: int) -> dict[str, Any]:
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "fact": row.fact,
        "source_refs": list(row.source_refs),
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "valid_until": row.valid_until.isoformat() if row.valid_until else None,
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
        "evidence_strength": row.evidence_strength,
        "is_attempted_failed_fix": row.predicate == "ATTEMPTED_FIX_FAILED",
        "verification_count": verifications,
    }


__all__ = ["PriorBugsReader"]
