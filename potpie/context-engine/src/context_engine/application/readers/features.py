"""FeaturesReader.

Answers "what does this repo/service provide?" from explicit feature claims.
It intentionally reads only PROVIDES / IMPLEMENTED_IN so feature context does
not get polluted by generic topology edges.
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


_FEATURE_PREDICATES: tuple[str, ...] = ("PROVIDES", "IMPLEMENTED_IN")


@dataclass(slots=True)
class FeaturesReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "features"

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
            "include_invalidated": req.include_invalidated,
            "as_of": req.as_of,
            "source_ref_in": req.source_refs,
            "limit": max(req.max_items * 6, 48),
        }
        if not anchors:
            return dedupe_claim_rows(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        **base,
                        predicate_in=_FEATURE_PREDICATES,
                        fact_query=req.query,
                    )
                )
            )

        rows: list[ClaimRow] = []
        rows.extend(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    **base,
                    predicate_in=("PROVIDES",),
                    subject_key_in=anchors,
                    fact_query=req.query,
                )
            )
        )
        rows.extend(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    **base,
                    predicate_in=("IMPLEMENTED_IN",),
                    object_key_in=anchors,
                    fact_query=req.query,
                )
            )
        )

        feature_keys = _feature_keys(rows)
        if feature_keys:
            rows.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        **base,
                        predicate_in=("PROVIDES",),
                        object_key_in=tuple(feature_keys),
                    )
                )
            )
            rows.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        **base,
                        predicate_in=("IMPLEMENTED_IN",),
                        subject_key_in=tuple(feature_keys),
                    )
                )
            )
        return dedupe_claim_rows(rows)


def _feature_keys(rows: Iterable[ClaimRow]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        if row.predicate == "PROVIDES":
            keys.add(row.object_key)
        elif row.predicate == "IMPLEMENTED_IN":
            keys.add(row.subject_key)
    return sorted(keys)


def _scope_overlap(row: ClaimRow, *, anchor_keys: Iterable[str]) -> float:
    if not anchor_keys:
        return 0.5
    anchors = set(anchor_keys)
    return 1.0 if row_in_anchor_set(row, anchors) else 0.6


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return claim_payload(row, extra={"properties": dict(row.properties or {})})


__all__ = ["FeaturesReader"]
