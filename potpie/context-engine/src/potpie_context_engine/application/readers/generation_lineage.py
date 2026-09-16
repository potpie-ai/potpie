"""GenerationLineageReader.

Walks provenance claims (GENERATED_FROM, IMPLEMENTS, IN_SESSION, and
related edges) so ``graph read --subgraph provenance --view lineage`` can
answer which prompt/spec/session produced a code span.
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
from potpie_context_engine.core.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_engine.domain.ranking import Candidate, RankingService

_PROVENANCE_PREDICATES = (
    "GENERATED_FROM",
    "IMPLEMENTS",
    "IN_SESSION",
    "DERIVED_FROM",
    "MODIFIES",
    "USED_CONTEXT",
)


@dataclass(slots=True)
class GenerationLineageReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "generation_lineage"

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = scoped_entity_keys(
            req.scope,
            prefixes=("code", "prompt", "spec", "session", "repo"),
            include_anchor_entity_key=True,
        )
        path = _scope_path(req.scope)
        rows = self._rows(req, anchor_keys=anchor_keys, path=path)

        candidates: list[Candidate] = []
        for row in rows:
            overlap = _scope_overlap(row, anchor_keys=anchor_keys, path=path)
            if (anchor_keys or path) and overlap == 0.0:
                continue
            candidates.append(
                Candidate(
                    candidate_key=claim_candidate_key(row),
                    payload=_payload_from_row(row),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    corroboration_count=claim_corroboration(row),
                    scope_overlap=overlap if (anchor_keys or path) else None,
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
            meta={
                "anchor_keys": list(anchor_keys),
                "path": path,
                "candidate_pool": len(rows),
            },
        )

    def _rows(
        self,
        req: ReadRequest,
        *,
        anchor_keys: Iterable[str],
        path: str | None,
    ) -> list[ClaimRow]:
        anchors = tuple(anchor_keys)
        base = {
            "pot_id": req.pot_id,
            "predicate_in": _PROVENANCE_PREDICATES,
            "include_invalidated": req.include_invalidated,
            "as_of": req.as_of,
            "valid_at_after": req.since,
            "valid_at_before": req.until,
            "source_ref_in": req.source_refs,
            # Path is not a ClaimQueryFilter axis, so collect the complete
            # candidate set before applying the local path predicate and the
            # reader's final max_items truncation.
            "limit": None if path else max(req.max_items * 8, 64),
            "fact_query": req.query,
        }

        def query_rows(filters: dict[str, Any]) -> list[ClaimRow]:
            rows = self.claim_query.find_claims(ClaimQueryFilter(**filters))
            # Vector-backed ports use a finite default when fact_query is set,
            # even with limit=None. For path-scoped reads, merge an unranked
            # lexical pass so a relevant path row cannot be hidden below that
            # semantic top-k cap; the reader ranks after local filtering.
            if path and req.query:
                lexical_filters = {**filters, "fact_query": None}
                rows.extend(
                    self.claim_query.find_claims(ClaimQueryFilter(**lexical_filters))
                )
            return rows

        rows = query_rows(base)
        if anchors:
            extra = query_rows({**base, "subject_key_in": anchors})
            extra += query_rows({**base, "object_key_in": anchors})
            rows = extra or rows
        rows = [row for row in rows if _row_in_window(row, req)]
        if path:
            rows = [
                row
                for row in rows
                if path in (row.subject_key or "")
                or path in (row.object_key or "")
                or path in str((row.properties or {}).get("path") or "")
                or path in str((row.properties or {}).get("file_path") or "")
            ]
        return dedupe_claim_rows(rows)


def _row_in_window(row: ClaimRow, req: ReadRequest) -> bool:
    """Apply the inclusive valid_at window defensively after backend reads."""
    if req.since is not None and (row.valid_at is None or row.valid_at < req.since):
        return False
    if req.until is not None and (row.valid_at is None or row.valid_at > req.until):
        return False
    return True


def _scope_path(scope: dict[str, Any]) -> str | None:
    for key in ("path", "file_path"):
        value = scope.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _scope_overlap(
    row: ClaimRow, *, anchor_keys: Iterable[str], path: str | None
) -> float:
    if anchor_keys and row_in_anchor_set(row, anchor_keys):
        return 1.0
    if path:
        blob = " ".join(
            (
                row.subject_key or "",
                row.object_key or "",
                str((row.properties or {}).get("path") or ""),
                str((row.properties or {}).get("file_path") or ""),
            )
        )
        if path in blob:
            return 0.8
    return 0.0


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return claim_payload(
        row,
        extra={
            "predicate": row.predicate,
            "properties": dict(row.properties or {}),
        },
    )


__all__ = ["GenerationLineageReader"]
