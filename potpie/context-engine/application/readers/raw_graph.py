"""RawGraphReader — the full canonical subgraph, for visualization.

Unlike the semantic P9 readers (which each filter to one use-case's predicate
set), this returns EVERY live ``:RELATES_TO`` claim in the pot — including the
generic ``RELATED_TO`` fallback edges that no semantic reader matches — so the
graph explorer can render the raw partition rather than a typed slice.

It is a visualization / debug read, not an agent retrieval family: the payload
shape matches :class:`InfraTopologyReader` (one item per edge:
``{predicate, subject_key, object_key, ...}``) so the same client transform
renders both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from application.readers._common import (
    ReadRequest,
    ReadResponse,
    coverage_status_from_count,
    rank_candidates,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ranking import Candidate, RankingService


@dataclass(slots=True)
class RawGraphReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "raw_graph"

    def read(self, req: ReadRequest) -> ReadResponse:
        rows = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=req.pot_id,
                include_invalidated=req.include_invalidated,
                as_of=req.as_of,
                limit=max(req.max_items * 4, 32),
            )
        )
        candidates = [
            Candidate(
                candidate_key=_candidate_key(row),
                payload=_payload_from_row(row),
                strength=row.evidence_strength,
                valid_at=row.valid_at,
                # No scoping for a raw dump — every edge is equally "in scope".
                scope_overlap=0.5,
                corroboration_count=_corroboration(row),
            )
            for row in rows
        ]
        ranked = rank_candidates(service=self.ranker, candidates=candidates, req=req)
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={"candidate_pool": len(rows)},
        )


def _candidate_key(row: ClaimRow) -> str:
    return f"{row.predicate}:{row.subject_key}:{row.object_key}:{row.source_ref or '-'}"


def _corroboration(row: ClaimRow) -> int:
    count = row.properties.get("corroboration_count")
    if isinstance(count, int) and count > 0:
        return count
    return 1


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "fact": row.fact,
        "environment": row.properties.get("environment"),
        "source_ref": row.source_ref,
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "evidence_strength": row.evidence_strength,
    }


__all__ = ["RawGraphReader"]
