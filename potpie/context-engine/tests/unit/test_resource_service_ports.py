"""Service behavior must hold for ANY store port implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.application.services.resource_service import ResourceService
from potpie_context_engine.testing_resources import (
    InMemoryResourceIndex,
    InMemoryResourceStore,
)
from potpie_context_core.ports.resource_store import ChunkWrite
from potpie_context_core.resource_models import ResourceManifest

pytestmark = pytest.mark.unit


class _SpyGraph:
    def __init__(self) -> None:
        self.mutations = []

    def mutate(self, request):
        self.mutations.append(request)

        class _R:
            ok = True
            status = "applied"
            claim_keys: tuple = ()
            detail = None

        return _R()


def _put_doc(service: ResourceService) -> None:
    manifest = ResourceManifest.model_validate(
        {
            "source_ref": "agent://x",
            "source_kind": "markdown",
            "sections": [
                {
                    "slug": "body",
                    "title": "Body",
                    "summary": "s",
                    "ordinal": 0,
                    "content_hash": "",
                    "chunks": [{"seq": 0, "label": "body"}],
                }
            ],
        }
    )
    service.store.put_document(
        pot_id="pot_x",
        doc_slug="doc-x",
        manifest=manifest,
        chunks=[ChunkWrite(section_slug="body", seq=0, content="hello world")],
    )


def test_remove_document_retracts_graph_claims_on_any_store() -> None:
    """A non-local store must not silently skip graph retraction on remove."""
    graph = _SpyGraph()
    service = ResourceService(
        store=InMemoryResourceStore(), index=InMemoryResourceIndex(), graph=graph
    )
    _put_doc(service)
    result = service.remove_document(pot_id="pot_x", doc_slug="doc-x")
    assert result["removed"] is True
    assert result["sections_retracted"] == ["body"]
    assert graph.mutations, "graph retraction was skipped for a non-local store"


def test_list_documents_payload_keys_are_stable_on_any_store() -> None:
    service = ResourceService(store=InMemoryResourceStore(), index=InMemoryResourceIndex())
    _put_doc(service)
    docs = service.list_documents(pot_id="pot_x")
    assert docs and set(docs[0]) == {
        "doc_slug",
        "source_ref",
        "source_kind",
        "revision",
        "updated_at",
        "section_count",
    }
