"""DocsReader FTS merge tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.application.readers._common import ReadRequest
from potpie_context_engine.application.readers.docs import DocsReader
from potpie_context_core.ports.claim_query import ClaimRow
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


def test_docs_reader_merges_fts_hits_with_claims() -> None:
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

    def chunk_search(pot_id: str, query: str, limit: int) -> list[dict]:
        assert pot_id == "pot-1"
        assert query == "install dependencies"
        return [
            {
                "doc_slug": "guide",
                "section_slug": "setup",
                "seq": 0,
                "score": -1.2,
                "uri": "potpie://res/guide/setup/0000",
                "content": "Install dependencies with pip",
                "provenance": [
                    {
                        "element_id": "elem-00001",
                        "page_number": 3,
                        "bbox": [0.1, 0.2, 0.9, 0.8],
                    }
                ],
            }
        ]

    reader = DocsReader(
        claim_query=store,
        ranker=RankingService(),
        chunk_search=chunk_search,
    )
    response = reader.read(
        ReadRequest(
            pot_id="pot-1",
            scope={"service": "context-engine"},
            query="install dependencies",
            max_items=5,
        )
    )

    assert response.items
    subject_keys = {
        item.candidate.payload.get("subject_key") for item in response.items
    }
    assert "document-section:guide/setup" in subject_keys
    assert response.meta.get("fts_hits") == 1
    fts_item = next(
        item
        for item in response.items
        if item.candidate.payload.get("subject_key") == "document-section:guide/setup"
    )
    props = fts_item.candidate.payload.get("properties") or {}
    assert props.get("provenance")[0]["page_number"] == 3
