"""DecisionsReader.

Surfaces ADR-style decision memory from DECIDED / AFFECTS claims. Decisions are
anchored as ``Decision -> scope`` edges, so scoped reads start at the object side
and then expand to sibling decision impact claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from context_engine.application.readers._common import (
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
from context_engine.domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from context_engine.domain.ranking import Candidate, RankingService


_DECISION_PREDICATES: tuple[str, ...] = ("DECIDED", "AFFECTS")


@dataclass(slots=True)
class DecisionsReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "decisions"

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = scoped_entity_keys(
            req.scope,
            prefixes=("service", "repo", "environment", "datastore", "cluster"),
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
            "predicate_in": _DECISION_PREDICATES,
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

        rows = self.claim_query.find_claims(
            ClaimQueryFilter(**base, object_key_in=anchors)
        )
        decision_keys = sorted({row.subject_key for row in rows})
        if decision_keys:
            rows.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        pot_id=req.pot_id,
                        predicate_in=_DECISION_PREDICATES,
                        subject_key_in=tuple(decision_keys),
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
    if row.subject_key.startswith("decision:"):
        return 0.6
    return 0.0


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return claim_payload(row, extra={"properties": dict(row.properties or {})})


__all__ = ["DecisionsReader"]
