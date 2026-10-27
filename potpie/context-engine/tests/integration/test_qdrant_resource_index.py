"""Run against an explicit Qdrant service; required by native-platform CI.

Each test has its own SQLite namespace and collection. Only that collection is
removed at teardown, so the configured service can safely contain other data.
"""

from __future__ import annotations

import dataclasses
import os
import threading

import httpx
import pytest

from potpie_context_engine.adapters.outbound.resources.index.qdrant_hybrid import (
    QdrantHybridResourceIndex,
)
from potpie_context_engine.testing.conformance import build_conformance_document


class Embedder:
    name = "qdrant-integration-v1"
    dimensions = 3
    calibrated = True

    def __init__(self):
        self.batches = 0

    def embed(self, text):
        return [1.0, float("quota" in text.lower()), 0.5]

    def embed_many(self, texts):
        self.batches += 1
        return [self.embed(text) for text in texts]


@pytest.fixture
def index(tmp_path):
    instance = QdrantHybridResourceIndex(home=tmp_path, embedder=Embedder())
    if not instance.capabilities().semantic:
        detail = instance.status().detail
        instance.close()
        if os.getenv("POTPIE_REQUIRE_QDRANT") == "1":
            pytest.fail(f"Required Qdrant test service is unavailable: {detail}")
        pytest.skip("Qdrant test service is unavailable")
    yield instance
    collection = instance.collection
    instance.close()
    headers = {"api-key": instance.qdrant_api_key} if instance.qdrant_api_key else {}
    with httpx.Client(timeout=5, trust_env=False, headers=headers) as client:
        response = client.delete(
            f"{instance.qdrant_url.rstrip('/')}/collections/{collection}"
        )
        assert response.status_code in {200, 404}


def add_document(index, pot="p", doc="guide", text=None):
    manifest, chunks = build_conformance_document(pot_id=pot, doc=doc)
    if text:
        manifest = dataclasses.replace(manifest, revision=2)
        chunks = tuple(
            dataclasses.replace(c, text=text, chars=len(text), revision=2)
            for c in chunks
        )
    return index.index_document(pot_id=pot, manifest=manifest, chunks=chunks)


def test_real_qdrant_lifecycle_filters_replacement_and_delete(index):
    add_document(index)
    add_document(index, pot="other", doc="private")
    assert index.drain().remaining == 0
    result = index.search(pot_id="p", doc="guide", query="quota")
    assert result.semantic_candidates == 3
    assert all(hit.doc == "guide" for hit in result.hits)
    add_document(index, text="completely replaced material")
    assert index.search(pot_id="p", query="quota").semantic_candidates == 0
    assert index.drain().remaining == 0
    index.drop_document(pot_id="p", slug="guide")
    assert not index.search(pot_id="p", query="quota").hits
    assert index.drain().remaining == 0
    assert index.search(pot_id="other", query="quota").semantic_candidates == 3


def test_real_qdrant_collection_loss_republishes_without_embedding(index):
    add_document(index)
    index.drain()
    assert index.embedder.batches == 1
    headers = {"api-key": index.qdrant_api_key} if index.qdrant_api_key else {}
    with httpx.Client(timeout=5, trust_env=False, headers=headers) as client:
        response = client.delete(
            f"{index.qdrant_url.rstrip('/')}/collections/{index.collection}"
        )
        assert response.status_code == 200
    report = index.drain()
    assert report.remaining == 0 and report.embedded == 0
    assert index.embedder.batches == 1
    assert index.search(pot_id="p", query="quota").semantic_candidates == 3


def test_real_qdrant_concurrent_writers_and_cleanup_without_embedder(index):
    add_document(index)
    other = QdrantHybridResourceIndex(home=index.home, embedder=Embedder())
    errors = []

    def drain(writer):
        try:
            report = writer.drain()
            assert report.detail is None
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=drain, args=(writer,)) for writer in (index, other)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        assert not errors and all(not thread.is_alive() for thread in threads)
        assert index.pending_count() == 0
        assert index.search(pot_id="p", query="quota").semantic_candidates == 3
    finally:
        other.close()
    cleanup = QdrantHybridResourceIndex(home=index.home, embedder=None)
    try:
        assert cleanup.drain_when_unavailable
        cleanup.drop_document(pot_id="p", slug="guide")
        assert cleanup.pending_count() == 5
        assert cleanup.drain().remaining == 0
        assert not index.search(pot_id="p", query="quota").hits
    finally:
        cleanup.close()
