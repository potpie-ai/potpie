"""OwnersReader.

Returns ownership claims for scoped services / repos plus team membership edges
that explain owner context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from potpie_context_engine.application.readers._common import (
    ReadRequest,
    ReadResponse,
    claim_candidate_key,
    claim_corroboration,
    claim_payload,
    claim_semantic_similarity,
    coverage_status_from_count,
    dedupe_claim_rows,
    rank_candidates,
    row_in_anchor_set,
    scoped_entity_keys,
)
from potpie_context_engine.domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from potpie_context_engine.domain.ranking import Candidate, RankingService


_OWNER_PREDICATES: tuple[str, ...] = ("OWNED_BY", "MEMBER_OF")


@dataclass(slots=True)
class OwnersReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "owners"

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = scoped_entity_keys(
            req.scope,
            prefixes=("service", "repo"),
            include_anchor_entity_key=True,
        )
        rows = self._rows(req, anchor_keys=anchor_keys)

        candidates: list[Candidate] = []
        for row in rows:
            overlap = _scope_overlap(row, anchor_keys=anchor_keys)
            if anchor_keys and overlap == 0.0:
                continue
            candidates.append(
                Candidate(
                    candidate_key=claim_candidate_key(row),
                    payload=_payload_from_row(row),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    corroboration_count=claim_corroboration(row),
                    scope_overlap=overlap if anchor_keys else None,
                    semantic_similarity=claim_semantic_similarity(row),
                )
            )

        ranked = rank_candidates(service=self.ranker, candidates=candidates, req=req)
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={"anchor_keys": list(anchor_keys), "candidate_pool": len(rows)},
        )

    def _rows(self, req: ReadRequest, *, anchor_keys: Iterable[str]) -> list[ClaimRow]:
        anchors = tuple(anchor_keys)
        base = {
            "pot_id": req.pot_id,
            "include_invalidated": req.include_invalidated,
            "as_of": req.as_of,
            "source_ref_in": req.source_refs,
            "limit": max(req.max_items * 8, 64),
            "fact_query": req.query,
        }
        if not anchors:
            return dedupe_claim_rows(
                self.claim_query.find_claims(
                    ClaimQueryFilter(**base, predicate_in=_OWNER_PREDICATES)
                )
            )

        rows = self.claim_query.find_claims(
            ClaimQueryFilter(**base, predicate_in=("OWNED_BY",), subject_key_in=anchors)
        )
        owner_keys = sorted({row.object_key for row in rows})
        if owner_keys:
            rows.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        pot_id=req.pot_id,
                        predicate_in=("MEMBER_OF",),
                        subject_key_in=tuple(
                            key for key in owner_keys if key.startswith("person:")
                        ),
                        include_invalidated=req.include_invalidated,
                        as_of=req.as_of,
                        source_ref_in=req.source_refs,
                        limit=max(req.max_items * 8, 64),
                    )
                )
            )
            rows.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        pot_id=req.pot_id,
                        predicate_in=("MEMBER_OF",),
                        object_key_in=tuple(
                            key for key in owner_keys if key.startswith("team:")
                        ),
                        include_invalidated=req.include_invalidated,
                        as_of=req.as_of,
                        source_ref_in=req.source_refs,
                        limit=max(req.max_items * 8, 64),
                    )
                )
            )
        return dedupe_claim_rows(rows)


def _scope_overlap(row: ClaimRow, *, anchor_keys: Iterable[str]) -> float:
    if not anchor_keys:
        return 0.5
    anchors = set(anchor_keys)
    if row_in_anchor_set(row, anchors):
        return 1.0
    if row.predicate == "MEMBER_OF":
        return 0.7
    return 0.0


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return claim_payload(row, extra={"properties": dict(row.properties or {})})


__all__ = ["OwnersReader"]
