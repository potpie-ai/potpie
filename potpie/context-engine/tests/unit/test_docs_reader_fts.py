"""DocsReader FTS merge tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.application.readers._common import ReadRequest
from potpie_context_engine.application.readers.docs import DocsReader
from potpie_context_engine.core.ports.claim_query import ClaimRow
from potpie_context_engine.domain.ranking import RankingService

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 5, 20, tzinfo=timezone.utc)


def _row(
    *,
    predicate: str,
    subject_key: str,
    object_key: str,
    fact: str = "",
) -> ClaimRow:
    return ClaimRow(
        pot_id="pot-1",
        predicate=predicate,
        subject_key=subject_key,
        object_key=object_key,
        fact=fact,
        valid_at=_NOW,
        evidence_strength="attested",
        source_system="agent",
    )


def test_docs_reader_prefers_answer_bearing_fts_hits_over_claims() -> None:
    store = InMemoryClaimQueryStore()
    store.set_entity_label(
        pot_id="pot-1", entity_key="document:other-doc", labels=("Document",)
    )
    store.add(
        _row(
            predicate="RELATED_TO",
            subject_key="document:other-doc",
            object_key="service:context-engine",
            fact="Unrelated document mention",
        )
    )

    def chunk_search(pot_id: str, query: str, limit: int, filters) -> list[dict]:
        assert pot_id == "pot-1"
        assert query == "install dependencies"
        assert filters.doc_slug == "guide"
        assert filters.source_kind == "pdf"
        assert filters.page_number == 3
        return [
            {
                "doc_slug": "guide",
                "section_slug": "setup",
                "seq": 0,
                "score": -1.2,
                "uri": "potpie://res/guide/setup/0000",
                "content": "Install dependencies with pip",
                "source_kind": "pdf",
                "page_numbers": [3],
                "provenance": [
                    {
                        "element_id": "elem-00001",
                        "page_number": 3,
                        "bbox": [0.1, 0.2, 0.9, 0.8],
                    }
                ],
                "citation": {
                    "chunk_uri": "potpie://res/guide/setup/0000",
                    "source_ref": "file:///guide.pdf",
                    "source_kind": "pdf",
                    "pages": [3],
                    "provenance_available": True,
                    "parser_tier": "docling",
                },
            },
            {
                "doc_slug": "guide",
                "section_slug": "setup",
                "seq": 1,
                "score": -1.4,
                "uri": "potpie://res/guide/setup/0001",
                "content": "Install optional dependencies for images",
                "source_kind": "pdf",
                "page_numbers": [3],
            },
        ]

    reader = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=chunk_search,
    )
    response = reader.read(
        ReadRequest(
            pot_id="pot-1",
            scope={
                "service": "context-engine",
                "doc_slug": "guide",
                "source_kind": "pdf",
                "page": "3",
            },
            query="install dependencies",
            max_items=5,
        )
    )

    assert response.items
    subject_keys = {
        item.candidate.payload.get("subject_key") for item in response.items
    }
    assert "docsection:guide:setup" in subject_keys
    assert "document:other-doc" not in subject_keys
    assert response.meta.get("fts_hits") == 2
    fts_item = next(
        item
        for item in response.items
        if item.candidate.payload.get("subject_key") == "docsection:guide:setup"
        and (item.candidate.payload.get("properties") or {}).get("provenance")
    )
    props = fts_item.candidate.payload.get("properties") or {}
    assert props.get("provenance")[0]["page_number"] == 3
    assert props.get("citation")["pages"] == [3]
    chunk_uris = {
        item.candidate.payload.get("chunk_uri")
        for item in response.items
        if item.candidate.payload.get("retrieval_source") == "fts"
    }
    assert chunk_uris == {
        "potpie://res/guide/setup/0000",
        "potpie://res/guide/setup/0001",
    }


def test_docs_reader_keeps_legacy_three_argument_chunk_search_compatible() -> None:
    store = InMemoryClaimQueryStore()

    def legacy_search(pot_id: str, query: str, limit: int) -> list[dict]:
        del limit
        assert (pot_id, query) == ("pot-1", "needle")
        return [
            {
                "doc_slug": "guide",
                "section_slug": "body",
                "seq": 0,
                "content": "needle body",
                "source_kind": "pdf",
                "page_numbers": [4],
            }
        ]

    reader = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=legacy_search,
    )
    response = reader.read(
        ReadRequest(
            pot_id="pot-1",
            query="needle",
            scope={"source_kind": "pdf", "page": "4"},
            max_items=5,
        )
    )

    assert [item.candidate.payload["chunk_uri"] for item in response.items] == [
        "potpie://res/guide/body/0000"
    ]


def test_docs_reader_preserves_raw_fts_relevance_metadata() -> None:
    store = InMemoryClaimQueryStore()

    def chunk_search(pot_id: str, query: str, limit: int, filters) -> list[dict]:
        del pot_id, query, limit, filters
        return [
            {
                "doc_slug": "guide",
                "section_slug": "body",
                "seq": 0,
                "score": -4.25,
                "content": "The answer-bearing passage.",
                "source_kind": "pdf",
                "page_numbers": [2],
            }
        ]

    response = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=chunk_search,
    ).read(ReadRequest(pot_id="pot-1", query="answer", max_items=5))

    item = response.items[0]
    properties = item.candidate.payload["properties"]
    assert properties["fts_score"] == -4.25
    assert properties["retrieval_source"] == "fts"


def test_docs_reader_propagates_structured_retrieval_metadata() -> None:
    store = InMemoryClaimQueryStore()

    def chunk_search(pot_id: str, query: str, limit: int, filters) -> list[dict]:
        del pot_id, query, limit, filters
        return [
            {
                "doc_slug": "paper",
                "section_slug": "table-10",
                "seq": 0,
                "score": -1.0,
                "content": "Table 10 GPT-2 accuracy 35.0%.",
                "evidence_text": "Table 10 GPT-2 accuracy 35.0%.",
                "facet_coverage": [0, 1],
                "structural_match": {
                    "score": 1.0,
                    "identifier_matches": 1,
                    "numeric_matches": 1,
                    "header_matches": 1,
                },
                "answer_bundle_complete": True,
            }
        ]

    response = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=chunk_search,
    ).read(ReadRequest(pot_id="pot-1", query="Table 10 accuracy", max_items=5))

    properties = response.items[0].candidate.payload["properties"]
    assert properties["evidence_text"].startswith("Table 10")
    assert properties["facet_coverage"] == [0, 1]
    assert properties["structural_match"]["identifier_matches"] == 1
    assert properties["answer_bundle_complete"] is True


def test_docs_reader_does_not_return_graph_summary_and_fts_duplicate() -> None:
    store = InMemoryClaimQueryStore()
    uri = "potpie://res/guide/body/0000"
    store.set_entity_label(
        pot_id="pot-1", entity_key="document-section:guide/body", labels=("DocumentSection",)
    )
    store.set_entity_label(
        pot_id="pot-1", entity_key="document:guide", labels=("Document",)
    )
    store.add(
        ClaimRow(
            pot_id="pot-1",
            predicate="RELATED_TO",
            subject_key="document-section:guide/body",
            object_key="document:guide",
            description="A section summary for the answer-bearing passage.",
            source_refs=("file:///guide.pdf", uri),
        )
    )

    def chunk_search(pot_id: str, query: str, limit: int, filters) -> list[dict]:
        del pot_id, query, limit, filters
        return [
            {
                "doc_slug": "guide",
                "section_slug": "body",
                "seq": 0,
                "score": -3.0,
                "uri": uri,
                "content": "The answer-bearing passage.",
                "source_kind": "pdf",
                "page_numbers": [2],
            }
        ]

    response = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=chunk_search,
    ).read(ReadRequest(pot_id="pot-1", query="answer", max_items=5))

    matching = [
        item
        for item in response.items
        if item.candidate.payload.get("chunk_uri") == uri
        or uri in item.candidate.payload.get("source_refs", [])
    ]
    assert len(matching) == 1
    assert matching[0].candidate.payload["retrieval_source"] == "fts"


def test_docs_reader_exposes_query_focused_evidence_snippet() -> None:
    store = InMemoryClaimQueryStore()
    answer = "The decisive answer is Wikipedia, removed to avoid overlap."

    def chunk_search(pot_id: str, query: str, limit: int, filters) -> list[dict]:
        del pot_id, query, limit, filters
        return [
            {
                "doc_slug": "guide",
                "section_slug": "body",
                "seq": 0,
                "score": -4.0,
                "content": ("lead text " * 80) + answer,
                "source_kind": "pdf",
                "page_numbers": [3],
            }
        ]

    response = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=chunk_search,
    ).read(ReadRequest(pot_id="pot-1", query="Wikipedia", max_items=5))

    properties = response.items[0].candidate.payload["properties"]
    assert answer in properties["content_snippet"]
    assert properties["content_snippet"] != properties["content_preview"]


def test_docs_reader_marks_missing_evidence_as_insufficient() -> None:
    response = DocsReader(
        claim_query=InMemoryClaimQueryStore(),
        ranker=RankingService(),
        chunk_search=lambda pot_id, query, limit, filters: [],
    ).read(ReadRequest(pot_id="pot-1", query="unsupported fact", max_items=5))

    assert response.meta["insufficient_evidence"] is True
    assert response.meta["match_mode"] == "none"


def test_docs_reader_abstains_on_generic_non_evidence_passage() -> None:
    store = InMemoryClaimQueryStore()

    def chunk_search(pot_id: str, query: str, limit: int, filters) -> list[dict]:
        del pot_id, query, limit, filters
        return [
            {
                "doc_slug": "guide",
                "section_slug": "body",
                "seq": 0,
                "score": 0.95,
                "retrieval_source": "fts",
                "content": "No.",
                "source_kind": "pdf",
                "page_numbers": [2],
            }
        ]

    response = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=chunk_search,
    ).read(
        ReadRequest(
            pot_id="pot-1",
            query="Which dataset was omitted from training?",
            max_items=5,
        )
    )

    assert response.items == ()
    assert response.meta["insufficient_evidence"] is True
