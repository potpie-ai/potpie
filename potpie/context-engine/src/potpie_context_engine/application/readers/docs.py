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

A read carrying a query then drops everything that scored far below the best
match in the pool — see :func:`_rows_clearing_relevance_floor`. A KNN returns
*k* rows whether or not any of them is an answer, so without a floor the reply
is padded to the limit with whatever the corpus happens to hold.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from collections.abc import Callable
import inspect
import re
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
    relative_relevance_floor,
    row_in_anchor_set,
    row_matches_query,
    scoped_entity_keys,
)
from potpie_context_engine.adapters.outbound.resources.graph_bridge import (
    section_entity_key,
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
from potpie_context_engine.core.ports.resource_store import RESOURCE_URI_PREFIX
from potpie_context_engine.core.ports.resource_index import ResourceSearchFilters
from potpie_context_engine.domain.ranking import Candidate, RankedItem, RankingService
from potpie_context_engine.domain.resource_models import chunk_uri


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
    chunk_search: Callable[..., list[dict[str, Any]]] | None = None

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = scoped_entity_keys(
            req.scope,
            prefixes=("service", "repo"),
            include_anchor_entity_key=True,
        )
        rows = _rows_clearing_relevance_floor(
            self._rows(req, anchor_keys=anchor_keys), req
        )

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

        resource_filters = _resource_filters(req.scope)
        candidates = [
            candidate
            for candidate, row in zip(candidates, rows, strict=False)
            if _row_matches_resource_filters(row, resource_filters)
        ]
        if _is_exact_metric_query(req.query or ""):
            # A graph summary can mention a benchmark family without carrying
            # the requested exact metric. Let the chunk index prove it.
            candidates = []

        fts_candidates = self._fts_candidates(req, resource_filters)
        use_fts = bool(fts_candidates) or (
            bool((req.query or "").strip()) and self.chunk_search is not None
        )
        ranking_req = (
            dataclasses.replace(req, max_items=0) if req.intent == "definition" else req
        )
        ranked = rank_candidates(
            service=self.ranker,
            candidates=fts_candidates if use_fts else candidates,
            req=ranking_req,
        )
        if req.intent == "definition":
            ranked = rank_definitions(ranked, req.query)
            if req.max_items > 0:
                ranked = ranked[: req.max_items]
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={
                "ranking_omitted": max(0, len(candidates) - len(ranked)),
                "candidate_pool_unit": "chunks" if use_fts else "claims",
                "anchor_keys": list(anchor_keys),
                "candidate_pool": len(rows),
                "fts_hits": len(fts_candidates),
                "match_mode": _match_mode(ranked),
                "insufficient_evidence": not ranked,
                **(
                    {
                        "definition_subject": definition_subject(
                            req.query, allow_bare=True
                        ),
                        "warnings": [
                            "Definition wording indicates relevance, not corroboration; verify the source refs and truth metadata, especially for competing expansions."
                        ],
                    }
                    if req.intent == "definition"
                    else {}
                ),
            },
        )

    def _fts_candidates(
        self, req: ReadRequest, filters: ResourceSearchFilters
    ) -> list[Candidate]:
        query = (req.query or "").strip()
        if not query or self.chunk_search is None:
            return []
        limit = max(req.max_items * 4, 32)
        if _accepts_filters(self.chunk_search):
            hits = self.chunk_search(req.pot_id, query, limit, filters)
        else:
            hits = self.chunk_search(req.pot_id, query, limit)  # type: ignore[call-arg]
        hits = [hit for hit in hits if _hit_matches_filters(hit, filters)]
        hits = [hit for hit in hits if _hit_supports_query(hit, query)]
        fts_scores = [
            float(hit.get("fts_score", hit["score"]))
            for hit in hits
            if isinstance(hit.get("fts_score", hit.get("score")), (int, float))
            and not isinstance(hit.get("fts_score", hit.get("score")), bool)
        ]
        candidates: list[Candidate] = []
        for hit in hits:
            doc_slug = str(hit.get("doc_slug") or "")
            section_slug = str(hit.get("section_slug") or "")
            seq = int(hit.get("seq") or 0)
            if not doc_slug or not section_slug:
                continue
            sec_key = section_entity_key(doc_slug, section_slug)
            uri = hit.get("uri") or chunk_uri(doc_slug, section_slug, seq)
            fts_score = hit.get("fts_score", hit.get("score"))
            relevance_score = hit.get("score")
            if isinstance(relevance_score, (int, float)) and 0.0 <= float(
                relevance_score
            ) <= 1.0 and hit.get("retrieval_source"):
                similarity = float(relevance_score)
            else:
                similarity = _fts_score_similarity(fts_score, fts_scores)
            properties: dict[str, Any] = {
                "doc_slug": doc_slug,
                "section_slug": section_slug,
                "seq": seq,
                "retrieval_source": hit.get("retrieval_source", "fts"),
            }
            for key in (
                "fts_score",
                "semantic_score",
                "citation",
                "source_ref",
                "source_kind",
                "section_title",
                "parser_tier",
                "page_numbers",
                "facet_coverage",
                "structural_match",
                "answer_bundle_complete",
                "provenance",
                "ocr_text",
            ):
                if key == "fts_score":
                    if fts_score is not None:
                        properties[key] = fts_score
                elif hit.get(key) is not None:
                    properties[key] = hit[key]
            if hit.get("content"):
                content = str(hit["content"])
                properties["content_preview"] = content[:500]
                properties["content_snippet"] = _evidence_snippet(content, query)
            if hit.get("evidence_text"):
                properties["evidence_text"] = str(hit["evidence_text"])
            if fts_score is not None:
                properties["relevance_score"] = similarity
            candidates.append(
                Candidate(
                    candidate_key=f"fts:{sec_key}:{seq}",
                    payload={
                        "subject_key": sec_key,
                        "subject_label": "DocumentSection",
                        "object_key": f"document:{doc_slug}",
                        "predicate": "RELATED_TO",
                        "chunk_uri": uri,
                        "retrieval_source": properties["retrieval_source"],
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
                    fact_query=query,
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
                    fact_query=query,
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


def _rows_clearing_relevance_floor(
    rows: list[ClaimRow], req: ReadRequest
) -> list[ClaimRow]:
    """Drop rows the query gives no reason to return.

    Without this the read hands back the ANN's top-k however distant every
    neighbor is, because a KNN always returns *k* rows — which is how a cloud
    cost spreadsheet came to answer a database-incident question, filling
    10 of 12 slots while the runbook section holding the literal error string
    ranked nowhere.

    Two rules, and the second is the one that matters:

    - A row survives on a **lexical** match even below the floor. An exact
      identifier is the strongest relevance signal there is and the weakest
      embedding signal — ``DBConnectionPoolExhausted`` is one token to a
      reader and noise to MiniLM.
    - A ``SECTION_OF`` row fetched by the one-hop expansion in :meth:`_rows`
      was never scored against the query — it is fetched by key, to carry the
      parent document and the chunk ids ``resource get`` takes. Judging it by a
      floor it never had a chance to clear would drop the section's fetchable
      ids on the floor with it, so it rides on its section instead. That is the
      same "structural rows hang off a section that already matched" rule
      :func:`_scope_overlap` applies, and unscoped reads are unaffected: there
      ``SECTION_OF`` *is* the queried row, it carries a score, and it is judged
      on it.

    ``req.query_threshold`` is deliberately not consulted by this claim reader.
    The former shared 0.70 default exceeded measured scores in the corpus, so
    the floor is derived from the pool instead. Explicit threshold support
    here remains separate from the resource index's passage threshold.
    """
    if not (req.query and req.query.strip()):
        return rows
    floor = relative_relevance_floor(rows)
    kept: list[ClaimRow] = []
    unscored_structure: list[ClaimRow] = []
    for row in rows:
        if (
            row.predicate.upper() == _STRUCTURE_PREDICATE
            and claim_semantic_similarity(row) is None
        ):
            unscored_structure.append(row)
        elif (
            req.intent == "definition"
            and has_definition(row.fact, definition_subject(req.query, allow_bare=True))
        ) or row_matches_query(row, req.query, threshold=floor):
            kept.append(row)
    # Admission and scoring are separate questions: a structural row is
    # admitted because its section survived, and scored from that section's
    # relevance *if it has one* — a section kept on a lexical match alone has
    # no similarity to lend, and that is no reason to drop its chunk ids.
    admitted = {row.subject_key for row in kept}
    relevance = _relevance_by_subject(kept)
    for row in unscored_structure:
        if row.subject_key not in admitted:
            continue
        similarity = relevance.get(row.subject_key)
        kept.append(row if similarity is None else _with_similarity(row, similarity))
    return kept


def _relevance_by_subject(rows: Iterable[ClaimRow]) -> dict[str, float]:
    """Best stamped similarity per subject among rows that were scored."""
    best: dict[str, float] = {}
    for row in rows:
        similarity = claim_semantic_similarity(row)
        if similarity is None:
            continue
        if similarity > best.get(row.subject_key, -1.0):
            best[row.subject_key] = similarity
    return best


def _with_similarity(row: ClaimRow, similarity: float) -> ClaimRow:
    """Lend a kept-on-behalf structural row its section's relevance.

    Without this it reaches the ranker with no similarity at all and takes the
    neutral 0.5 default, which on a corpus whose real scores top out near 0.6
    means an unscored row outranks every measured one — "never compared to the
    query" beating "compared and matched". It was admitted because its section
    matched, so it is scored the same way.
    """
    return dataclasses.replace(
        row, properties={**dict(row.properties), "semantic_similarity": similarity}
    )


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


_EVIDENCE_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have how in into "
    "is it its no not of on or than that the their then there these this those "
    "to was were what when where which who why will with would".split()
)
_EVIDENCE_SYNONYMS = {
    "omitted": {"removed", "excluded"},
    "excluded": {"removed", "omitted"},
    "prevent": {"avoid", "block"},
    "approximately": {"about", "roughly"},
}


def _is_exact_metric_query(query: str) -> bool:
    lowered = query.casefold()
    return ("exact" in lowered or "p-value" in lowered) and any(
        term in lowered
        for term in ("accuracy", "score", "p-value", "perplexity", "bleu", "f1")
    )


def _match_mode(items: list[RankedItem]) -> str:
    if not items:
        return "none"
    sources = {
        str(item.candidate.payload.get("retrieval_source") or "graph")
        for item in items
    }
    if "hybrid" in sources or len(sources) > 1:
        return "hybrid"
    return next(iter(sources))


def _resource_filters(scope: Any) -> ResourceSearchFilters:
    doc_slug = scope.get("doc_slug") or scope.get("document")
    source_kind = scope.get("source_kind") or scope.get("document_type")
    raw_page = scope.get("page_number") or scope.get("page")
    page_number: int | None = None
    if raw_page not in (None, ""):
        try:
            page_number = int(raw_page)
        except (TypeError, ValueError) as exc:
            raise ValueError("document page filter must be a positive integer") from exc
    return ResourceSearchFilters(
        doc_slug=str(doc_slug) if doc_slug else None,
        source_kind=str(source_kind) if source_kind else None,
        page_number=page_number,
    )


def _accepts_filters(search: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(search).parameters.values()
    except (TypeError, ValueError):
        return True
    positional = 0
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= 4


def _hit_matches_filters(hit: dict[str, Any], filters: ResourceSearchFilters) -> bool:
    if filters.doc_slug is not None and hit.get("doc_slug") != filters.doc_slug:
        return False
    if filters.source_kind is not None and hit.get("source_kind") != filters.source_kind:
        return False
    if filters.page_number is not None:
        pages = hit.get("page_numbers") or []
        if filters.page_number not in pages:
            return False
    return True


def _row_matches_resource_filters(
    row: ClaimRow, filters: ResourceSearchFilters
) -> bool:
    if filters.doc_slug is not None:
        doc_slugs: set[str] = set()
        for key in (row.subject_key, row.object_key):
            if key.startswith("document:"):
                doc_slugs.add(key.removeprefix("document:"))
            elif key.startswith("document-section:"):
                doc_slugs.add(key.removeprefix("document-section:").split("/", 1)[0])
        if filters.doc_slug not in doc_slugs:
            return False
    properties = row.properties or {}
    if filters.source_kind is not None and properties.get("source_kind") != filters.source_kind:
        return False
    if filters.page_number is not None:
        pages = properties.get("page_numbers")
        if pages is None and properties.get("page_number") is not None:
            pages = [properties["page_number"]]
        if not isinstance(pages, (list, tuple, set)):
            return False
        if filters.page_number not in pages:
            return False
    return True


def _fts_score_similarity(score: Any, scores: list[float]) -> float:
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not scores:
        return 0.5
    best = min(scores)
    worst = max(scores)
    if best == worst:
        return 1.0
    return max(0.0, min(1.0, (worst - float(score)) / (worst - best)))


def _evidence_snippet(content: str, query: str, *, limit: int = 500) -> str:
    if len(content) <= limit:
        return content
    tokens = [token for token in re.findall(r"[\w-]+", query) if len(token) >= 2]
    lowered = content.casefold()
    match_start = min(
        (position for token in tokens if (position := lowered.find(token.casefold())) >= 0),
        default=-1,
    )
    if match_start < 0:
        return content[: limit - 3].rstrip() + "..."
    window = max(80, limit // 3)
    start = max(0, match_start - window)
    end = min(len(content), start + limit)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    available = limit - len(prefix) - len(suffix)
    if end == len(content):
        start = max(0, len(content) - available)
    else:
        start = max(0, min(start, len(content) - available))
        end = start + available
    return f"{prefix}{content[start:end].strip()}{suffix}"


def _query_evidence_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[\w-]+", query.casefold()):
        if len(token) < 2 or token in _EVIDENCE_STOPWORDS:
            continue
        terms.add(token)
        if token.endswith("s") and len(token) > 3:
            terms.add(token[:-1])
        terms.update(_EVIDENCE_SYNONYMS.get(token, ()))
    return terms


def _hit_supports_query(hit: dict[str, Any], query: str) -> bool:
    source = str(hit.get("retrieval_source") or "fts").lower()
    if source == "semantic":
        score = hit.get("semantic_score", hit.get("score"))
        return isinstance(score, (int, float)) and float(score) >= 0.25
    searchable = " ".join(
        str(hit.get(key) or "") for key in ("content", "section_title", "ocr_text")
    ).casefold()
    if not searchable.strip():
        return True
    if re.search(r"\b(title|heading|name)\b", query, flags=re.IGNORECASE):
        return bool(str(hit.get("section_title") or "").strip())
    return any(term in searchable for term in _query_evidence_terms(query))


__all__ = ["DocsReader"]
