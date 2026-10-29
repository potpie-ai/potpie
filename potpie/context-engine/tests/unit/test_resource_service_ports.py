"""Service behavior must hold for ANY store port implementation."""

from __future__ import annotations

import pytest

from potpie_context_engine.application.services.resource_service import ResourceService
from potpie_context_engine.testing_resources import (
    InMemoryResourceIndex,
    InMemoryResourceStore,
)
from potpie_context_engine.core.ports.resource_store import ChunkWrite
from potpie_context_engine.core.resource_models import ResourceManifest

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
    service = ResourceService(
        store=InMemoryResourceStore(), index=InMemoryResourceIndex()
    )
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


def test_search_keeps_legacy_index_signature_and_filters_enriched_hits() -> None:
    class _LegacyIndex(InMemoryResourceIndex):
        def search(self, *, pot_id: str, query: str, limit: int = 20):
            return super().search(pot_id=pot_id, query=query, limit=limit)

    index = _LegacyIndex()
    service = ResourceService(store=InMemoryResourceStore(), index=index)
    _put_doc(service)
    index.index_document(
        pot_id="pot_x",
        doc_slug="doc-x",
        chunks=service.store.iter_chunks(pot_id="pot_x", doc_slug="doc-x"),
    )

    assert service.search_chunks(pot_id="pot_x", query="hello", source_kind="markdown")
    assert service.search_chunks(pot_id="pot_x", query="hello", source_kind="pdf") == []


def test_index_failure_rolls_back_local_store_and_keeps_previous_search_hit(
    tmp_path,
) -> None:
    from pathlib import Path

    from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
        LocalResourceStore,
    )
    from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
        parse_file_to_staging,
    )

    store = LocalResourceStore(home=Path(tmp_path) / "home")
    index = InMemoryResourceIndex()
    service = ResourceService(store=store, index=index)

    first = Path(tmp_path) / "first.md"
    first.write_text("## Guide\n\nThe old evidence remains available.", encoding="utf-8")
    first_stage = Path(tmp_path) / "first-stage"
    parse_file_to_staging(first, first_stage)
    assert not service.import_staging(
        pot_id="pot-x", doc_slug="guide", staging_dir=first_stage
    ).errors

    class _FailingIndex:
        def capabilities(self):
            return index.capabilities()

        def index_document(self, *, pot_id, doc_slug, chunks):
            del pot_id, doc_slug, chunks
            raise RuntimeError("injected index failure")

        def remove_document(self, *, pot_id, doc_slug):
            index.remove_document(pot_id=pot_id, doc_slug=doc_slug)

        def search(self, *, pot_id, query, limit=20, filters=None):
            return index.search(
                pot_id=pot_id, query=query, limit=limit, filters=filters
            )

        def status(self, pot_id=None):
            return index.status(pot_id)

    service.index = _FailingIndex()
    second = Path(tmp_path) / "second.md"
    second.write_text("## Guide\n\nThe new evidence must not replace old evidence.", encoding="utf-8")
    second_stage = Path(tmp_path) / "second-stage"
    parse_file_to_staging(second, second_stage)

    failed = service.import_staging(
        pot_id="pot-x", doc_slug="guide", staging_dir=second_stage, force=True
    )

    assert failed.errors[0]["code"] == "resource_index_failed"
    assert service.get_chunk(
        pot_id="pot-x", uri="potpie://res/guide/guide/0000"
    )["content"].startswith("The old evidence")
    assert service.search_chunks(pot_id="pot-x", query="old evidence")


def test_graph_failure_is_retryable_without_recording_file_hash_or_reparsing(
    tmp_path, monkeypatch
) -> None:
    from pathlib import Path

    from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
        LocalResourceStore,
    )
    from potpie_context_engine.adapters.outbound.resources.parsers import dispatch

    class _Graph:
        def __init__(self) -> None:
            self.calls = 0

        def mutate(self, request):
            del request
            self.calls += 1

            class _Result:
                ok = self.calls > 1
                status = "applied" if self.calls > 1 else "error"
                claim_keys = ("claim:guide",)
                detail = "graph temporarily unavailable" if self.calls == 1 else None

            return _Result()

    source = Path(tmp_path) / "guide.md"
    source.write_text("## Guide\n\nRetryable graph evidence.", encoding="utf-8")
    store = LocalResourceStore(home=Path(tmp_path) / "home")
    graph = _Graph()
    service = ResourceService(store=store, graph=graph)

    first = service.ingest_file(pot_id="pot-x", doc_slug="guide", source_path=source)
    assert first.errors[0]["code"] == "resource_graph_write_failed"
    assert not store.is_file_imported(
        pot_id="pot-x", content_hash=dispatch.file_sha256(source)
    )

    original_parser = dispatch.parse_file_to_staging

    def _unexpected_reparse(*args, **kwargs):
        del args, kwargs
        raise AssertionError("retry should reuse the pending staging tree")

    monkeypatch.setattr(
        "potpie_context_engine.application.services.resource_service.parse_file_to_staging",
        _unexpected_reparse,
    )
    second = service.ingest_file(pot_id="pot-x", doc_slug="guide", source_path=source)
    monkeypatch.setattr(
        "potpie_context_engine.application.services.resource_service.parse_file_to_staging",
        original_parser,
    )
    assert not second.errors
    assert second.graph_written
    assert store.is_file_imported(
        pot_id="pot-x", content_hash=dispatch.file_sha256(source)
    )
    with pytest.raises(ValueError, match="already imported"):
        service.ingest_file(pot_id="pot-x", doc_slug="guide", source_path=source)
