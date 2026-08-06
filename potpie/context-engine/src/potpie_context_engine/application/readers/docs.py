"""DocsReader.

Returns reference-material claims: an ingested ``Document`` or one of its
``DocumentSection``s pointed at a scope through ``DOCUMENTS``, plus the older
``Document RELATED_TO scope`` fallback claims that ``doc_reference`` records
still land on. Sections that match are expanded one hop over ``SECTION_OF`` so
a result carries the document it came from.

``ClaimQueryFilter.subject_label`` takes a single label, so the two subject
labels are two queries merged by :func:`dedupe_claim_rows`. Where a document
carries both spellings for one scope, the typed one wins and the fallback is
dropped, so the document takes one result slot rather than two.

An unscoped read also returns the ``SECTION_OF`` claims themselves — the rows
``resource import`` writes, whose fact is the section's summary and whose source
refs are that section's chunk ids. That is what makes a search land on a section
and come back already holding the ids ``resource get`` takes, with no ``list``
hop in between.
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
from potpie_context_core.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_core.ports.resource_store import RESOURCE_URI_PREFIX
from potpie_context_engine.domain.ranking import Candidate, RankingService


_DOCUMENT_LABEL = "Document"
_SECTION_LABEL = "DocumentSection"
_DOC_SUBJECT_LABELS: tuple[str, ...] = (_DOCUMENT_LABEL, _SECTION_LABEL)
# ``DOCUMENTS`` is the typed reference-material predicate; ``RELATED_TO`` stays
# so doc claims written before the predicate existed keep reading.
_DOCUMENTS_PREDICATE = "DOCUMENTS"
_LEGACY_PREDICATE = "RELATED_TO"
_DOC_PREDICATES: tuple[str, ...] = (_DOCUMENTS_PREDICATE, _LEGACY_PREDICATE)
_STRUCTURE_PREDICATE = "SECTION_OF"


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
        base: dict[str, Any] = {
            "pot_id": req.pot_id,
            "include_invalidated": req.include_invalidated,
            "as_of": req.as_of,
            "source_ref_in": req.source_refs,
            "limit": max(req.max_items * 8, 64),
        }
        scoped: dict[str, Any] = {"object_key_in": anchors} if anchors else {}

        rows: list[ClaimRow] = []
        section_keys: set[str] = set()
        for label in _DOC_SUBJECT_LABELS:
            found = self.claim_query.find_claims(
                ClaimQueryFilter(
                    **base,
                    **scoped,
                    predicate_in=_DOC_PREDICATES,
                    subject_label=label,
                    fact_query=req.query,
                )
            )
            if label == _SECTION_LABEL:
                section_keys.update(row.subject_key for row in found)
            rows.extend(found)

        if not anchors:
            # ``SECTION_OF`` is a section's own claim — it carries the
            # agent-authored summary as its fact and that section's chunk ids as
            # its source refs, which is the whole search-then-get path (R13).
            # Only unscoped: its object is the parent document, so it can never
            # match an ``object_key_in`` filter naming a service or repo, and
            # asking for it under a scope would just cost a query that returns
            # nothing.
            sections = self.claim_query.find_claims(
                ClaimQueryFilter(
                    **base,
                    predicate_in=(_STRUCTURE_PREDICATE,),
                    subject_label=_SECTION_LABEL,
                    fact_query=req.query,
                )
            )
            section_keys.update(row.subject_key for row in sections)
            rows.extend(sections)

        if section_keys:
            rows.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        **base,
                        predicate_in=(_STRUCTURE_PREDICATE,),
                        subject_key_in=tuple(sorted(section_keys)),
                    )
                )
            )
        return _collapse_legacy_duplicates(dedupe_claim_rows(rows))


def _collapse_legacy_duplicates(rows: list[ClaimRow]) -> list[ClaimRow]:
    """Drop a ``RELATED_TO`` row a ``DOCUMENTS`` row already says.

    Nothing migrates the fallback: ``doc_reference`` records still emit
    ``RELATED_TO`` while an agent can write ``DOCUMENTS`` for the same pair, and
    ``dedupe_claim_rows`` keys on the predicate, so both survive. Left alone
    they are two ranked candidates for one document — two of the agent's
    ``max_items`` spent saying the same thing.
    """
    typed = {
        (row.subject_key, row.object_key)
        for row in rows
        if row.predicate.upper() == _DOCUMENTS_PREDICATE
    }
    if not typed:
        return rows
    return [
        row
        for row in rows
        if row.predicate.upper() != _LEGACY_PREDICATE
        or (row.subject_key, row.object_key) not in typed
    ]


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


__all__ = ["DocsReader"]
