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

from potpie_context_engine.application.readers._common import (
    ReadRequest,
    ReadResponse,
    claim_candidate_key,
    claim_corroboration,
    claim_payload,
    coverage_status_from_count,
    dedupe_claim_rows,
    rank_candidates,
)
from potpie_context_engine.domain.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_engine.domain.ranking import Candidate, RankingService


@dataclass(slots=True)
class RawGraphReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "raw_graph"

    def read(self, req: ReadRequest) -> ReadResponse:
        rows = dedupe_claim_rows(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    source_ref_in=req.source_refs,
                    limit=max(req.max_items * 4, 32),
                )
            )
        )
        candidates = [
            Candidate(
                candidate_key=claim_candidate_key(row),
                payload=_payload_from_row(row),
                strength=row.evidence_strength,
                valid_at=row.valid_at,
                # No scoping for a raw dump — every edge is equally "in scope".
                scope_overlap=0.5,
                corroboration_count=claim_corroboration(row),
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


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return claim_payload(row)


__all__ = ["RawGraphReader"]
