"""DocsReader.

Returns document-reference claims. The current write path stores doc_reference
records as ``Document RELATED_TO scope`` fallback claims, so this reader keeps
the slice narrow to Document-subject RELATED_TO edges.

When a query is present and ``chunk_search`` is wired, FTS5 chunk hits are
merged with graph claim candidates via reciprocal rank fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable

from potpie_context_engine.adapters.outbound.resources.graph_bridge import section_entity_key
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
    relative_relevance_floor,
    row_in_anchor_set,
    row_matches_query,
    scoped_entity_keys,
)
from potpie_context_engine.core.definition_query import definition_subject
from potpie_context_engine.domain.definition_retrieval import (
    has_definition,
    rank_definitions,
)
from potpie_context_engine.core.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_engine.domain.ranking import Candidate, RankedItem, RankingService
from potpie_context_engine.domain.resource_models import chunk_uri

ChunkSearchFn = Callable[[str, str, int], list[dict[str, Any]]]


_DOCUMENT_LABEL = "Document"
_SECTION_LABEL = "DocumentSection"
_DOC_SUBJECT_LABELS: tuple[str, ...] = (_DOCUMENT_LABEL, _SECTION_LABEL)
_DOC_PREDICATES: tuple[str, ...] = ("DOCUMENTS",)
_STRUCTURE_PREDICATE = "SECTION_OF"


@dataclass(slots=True)
class DocsReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "docs"
    chunk_search: ChunkSearchFn | None = None

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = scoped_entity_keys(
            req.scope,
            prefixes=("service", "repo"),
            include_anchor_entity_key=True,
        )
        rows = _rows_clearing_relevance_floor(
            self._rows(req, anchor_keys=anchor_keys), req
        )

        claim_candidates: list[Candidate] = []
        for row in rows:
            overlap = _scope_overlap(row, anchor_keys=anchor_keys)
            if anchor_keys and overlap == 0.0:
                continue
            claim_candidates.append(
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

        fts_candidates = self._fts_candidates(req)
        if fts_candidates and claim_candidates:
            claim_ranked = rank_candidates(
                service=self.ranker, candidates=claim_candidates, req=req
            )
            fts_ranked = rank_candidates(
                service=self.ranker, candidates=fts_candidates, req=req
            )
            ranked = _rrf_merge_ranked(
                [claim_ranked, fts_ranked], max_items=req.max_items
            )
        elif fts_candidates:
            ranked = rank_candidates(
                service=self.ranker, candidates=fts_candidates, req=req
            )
        else:
            ranked = rank_candidates(
                service=self.ranker, candidates=claim_candidates, req=req
            )

        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={
                "anchor_keys": list(anchor_keys),
                "candidate_pool": len(rows),
                "fts_hits": len(fts_candidates),
            },
        )

    def _fts_candidates(self, req: ReadRequest) -> list[Candidate]:
        query = (req.query or "").strip()
        if not query or self.chunk_search is None:
            return []
        limit = max(req.max_items * 4, 32)
        hits = self.chunk_search(req.pot_id, query, limit)
        candidates: list[Candidate] = []
        for rank, hit in enumerate(hits):
            doc_slug = str(hit.get("doc_slug") or "")
            section_slug = str(hit.get("section_slug") or "")
            seq = int(hit.get("seq") or 0)
            if not doc_slug or not section_slug:
                continue
            sec_key = section_entity_key(doc_slug, section_slug)
            uri = hit.get("uri") or chunk_uri(doc_slug, section_slug, seq)
            similarity = _fts_rank_similarity(rank, len(hits))
            properties: dict[str, Any] = {
                "doc_slug": doc_slug,
                "section_slug": section_slug,
                "seq": seq,
            }
            if hit.get("content"):
                content = str(hit["content"])
                properties["content_preview"] = content[:500]
            if hit.get("provenance") is not None:
                properties["provenance"] = hit["provenance"]
            if hit.get("ocr_text"):
                properties["ocr_text"] = hit["ocr_text"]
            candidates.append(
                Candidate(
                    candidate_key=f"fts:{sec_key}:{seq}",
                    payload={
                        "subject_key": sec_key,
                        "subject_label": "DocumentSection",
                        "object_key": f"document:{doc_slug}",
                        "predicate": "RELATED_TO",
                        "chunk_uri": uri,
                        "retrieval_source": "fts",
                        "properties": properties,
                    },
                    strength="stated",
                    semantic_similarity=similarity,
                    scope_overlap=0.5,
                )
            )
        return candidates

    def _rows(self, req: ReadRequest, *, anchor_keys: Iterable[str]) -> list[ClaimRow]:
        anchors = tuple(anchor_keys)
        # Definition wording is a routing cue; the term itself is the recall query.
        query = (
            (definition_subject(req.query, allow_bare=True) or req.query)
            if req.intent == "definition"
            else req.query
        )
        base: dict[str, Any] = {
            "pot_id": req.pot_id,
            "predicate_in": ("RELATED_TO",),
            "include_invalidated": req.include_invalidated,
            "as_of": req.as_of,
            "source_ref_in": req.source_refs,
            "limit": max(req.max_items * 8, 64),
        }
        all_rows: list[ClaimRow] = []
        for label in ("Document", "DocumentSection"):
            filt: dict[str, Any] = {**base, "subject_label": label}
            if anchors:
                all_rows.extend(
                    self.claim_query.find_claims(
                        ClaimQueryFilter(**filt, object_key_in=anchors)
                    )
                )
            else:
                all_rows.extend(self.claim_query.find_claims(ClaimQueryFilter(**filt)))
        return dedupe_claim_rows(all_rows)


def _scope_overlap(row: ClaimRow, *, anchor_keys: Iterable[str]) -> float:
    if not anchor_keys:
        return 0.5
    if row_in_anchor_set(row, anchor_keys):
        return 1.0
    # Structural rows hang off a section that already matched the scope, so
    # they stay in the pool at a reduced weight instead of being dropped.
    if row.predicate.upper() == _STRUCTURE_PREDICATE:
        return 0.6
    return 0.0


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    # ``chunk_ids`` is the subset of source refs that ``resource get`` accepts.
    # It is pulled out by name so an agent does not have to know that a
    # ``potpie://res/`` string among the refs is the fetchable one.
    refs = tuple(row.source_refs or ((row.source_ref,) if row.source_ref else ()))
    chunk_ids = [ref for ref in refs if ref and ref.startswith(RESOURCE_URI_PREFIX)]
    extra: dict[str, Any] = {"properties": dict(row.properties or {})}
    if chunk_ids:
        extra["chunk_ids"] = chunk_ids
    return claim_payload(row, extra=extra)


def _fts_rank_similarity(rank: int, total: int) -> float:
    if total <= 0:
        return 0.5
    return max(0.55, 1.0 - (rank / max(total, 1)) * 0.45)


def _subject_key_from_ranked(item: RankedItem) -> str:
    payload = item.candidate.payload
    key = payload.get("subject_key")
    if key:
        return str(key)
    return item.candidate.candidate_key


def _rrf_merge_ranked(
    lists: list[list[RankedItem]],
    *,
    max_items: int,
    k: int = 60,
) -> list[RankedItem]:
    scores: dict[str, float] = {}
    best: dict[str, RankedItem] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            key = _subject_key_from_ranked(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best or item.score > best[key].score:
                best[key] = item
    ordered = sorted(scores.keys(), key=lambda sk: scores[sk], reverse=True)
    if max_items > 0:
        ordered = ordered[:max_items]
    return [best[sk] for sk in ordered]


__all__ = ["DocsReader"]
