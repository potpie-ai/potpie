"""DocsReader.

Returns document-reference claims. The current write path stores doc_reference
records as ``Document RELATED_TO scope`` fallback claims, so this reader keeps
the slice narrow to Document-subject RELATED_TO edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from application.readers._common import (
    ReadRequest,
    ReadResponse,
    claim_candidate_key,
    claim_corroboration,
    claim_payload,
    coverage_status_from_count,
    dedupe_claim_rows,
    rank_candidates,
    row_in_anchor_set,
    scoped_entity_keys,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ranking import Candidate, RankingService


@dataclass(slots=True)
class DocsReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "docs"

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
            sim = row.properties.get("semantic_similarity")
            candidates.append(
                Candidate(
                    candidate_key=claim_candidate_key(row),
                    payload=_payload_from_row(row),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    corroboration_count=claim_corroboration(row),
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
            meta={"anchor_keys": list(anchor_keys), "candidate_pool": len(rows)},
        )

    def _rows(self, req: ReadRequest, *, anchor_keys: Iterable[str]) -> list[ClaimRow]:
        anchors = tuple(anchor_keys)
        base = {
            "pot_id": req.pot_id,
            "predicate_in": ("RELATED_TO",),
            "subject_label": "Document",
            "include_invalidated": req.include_invalidated,
            "as_of": req.as_of,
            "source_ref_in": req.source_refs,
            "limit": max(req.max_items * 8, 64),
            "fact_query": req.query,
        }
        if not anchors:
            return dedupe_claim_rows(
                self.claim_query.find_claims(ClaimQueryFilter(**base))
            )
        return dedupe_claim_rows(
            self.claim_query.find_claims(
                ClaimQueryFilter(**base, object_key_in=anchors)
            )
        )


def _scope_overlap(row: ClaimRow, *, anchor_keys: Iterable[str]) -> float:
    if not anchor_keys:
        return 0.5
    return 1.0 if row_in_anchor_set(row, anchor_keys) else 0.0


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return claim_payload(row, extra={"properties": dict(row.properties or {})})


__all__ = ["DocsReader"]
