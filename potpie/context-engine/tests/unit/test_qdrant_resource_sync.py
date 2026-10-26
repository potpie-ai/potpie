"""Projection consistency under outages, retries, replacement and concurrency."""

from __future__ import annotations

import dataclasses
import math
import sqlite3
import threading

import pytest

from potpie_context_engine.adapters.outbound.resources.index.qdrant_hybrid import (
    QdrantHybridResourceIndex,
)
from potpie_context_engine.adapters.outbound.resources.index.qdrant_http import (
    QdrantError,
)
from potpie_context_engine.adapters.outbound.resources.index.sqlite_fts import (
    SqliteFtsResourceIndex,
)
from potpie_context_engine.adapters.outbound.resources.index.qdrant_sync import (
    pack_vector,
)
from potpie_context_engine.testing.conformance import build_conformance_document


class Embedder:
    name = "test-v1"
    dimensions = 3
    calibrated = True

    def __init__(self):
        self.calls = 0
        self.on_embed = None

    def embed(self, text):
        return [1.0, 1.0 if "quota" in text.lower() else 0.0, 0.5]

    def embed_many(self, texts):
        self.calls += 1
        vectors = [self.embed(text) for text in texts]
        if self.on_embed:
            action, self.on_embed = self.on_embed, None
            action()
        return vectors


class FakeQdrant:
    """A stateful remote double: errors can happen after the remote commit."""

    def __init__(self):
        self.collections = {}
        self.offline = False
        self.lose_upsert_ack = False
        self.lose_create_ack = False
        self.lose_delete_ack = False
        self.upserts = []
        self.queries = []

    def check(self):
        if self.offline:
            raise QdrantError("Qdrant service could not be reached")

    def ensure_collection(self, name, dimensions, *, on_missing=None):
        self.check()
        missing = name not in self.collections
        if missing:
            if on_missing:
                on_missing()
            self.collections[name] = {}
            if self.lose_create_ack:
                self.lose_create_ack = False
                raise QdrantError("Qdrant request timed out")
        return missing

    def upsert(self, collection, points):
        self.check()
        if points:
            self.upserts.append([p["id"] for p in points])
        self.collections[collection].update({p["id"]: p for p in points})
        if self.lose_upsert_ack:
            self.lose_upsert_ack = False
            raise QdrantError("Qdrant request timed out")

    def delete(self, collection, point_ids):
        self.check()
        for point_id in point_ids:
            self.collections.get(collection, {}).pop(point_id, None)
        if self.lose_delete_ack:
            self.lose_delete_ack = False
            raise QdrantError("Qdrant request timed out")

    def query(self, collection, vector, filter, limit, offset=0):
        self.check()
        self.queries.append((filter, offset))
        points = [
            p
            for p in self.collections[collection].values()
            if all(
                p["payload"].get(c["key"]) == c["match"]["value"]
                for c in filter["must"]
            )
        ]

        def cosine(other):
            return sum(a * b for a, b in zip(vector, other)) / math.sqrt(
                sum(a * a for a in vector) * sum(b * b for b in other)
            )

        hits = [
            {"id": p["id"], "score": cosine(p["vector"]), "payload": p["payload"]}
            for p in points
        ]
        return sorted(hits, key=lambda h: -h["score"])[offset : offset + limit]


def make_index(tmp_path, remote=None, embedder=None):
    return QdrantHybridResourceIndex(
        home=tmp_path, transport=remote or FakeQdrant(), embedder=embedder or Embedder()
    )


def import_doc(index, *, pot="p", doc="guide", text=None):
    manifest, chunks = build_conformance_document(pot_id=pot, doc=doc)
    if text:
        chunks = tuple(
            dataclasses.replace(c, text=text, chars=len(text), revision=2)
            for c in chunks
        )
        manifest = dataclasses.replace(manifest, revision=2)
    return index.index_document(pot_id=pot, manifest=manifest, chunks=chunks)


def test_offline_then_recovery_keeps_lexical_and_reports_backlog(tmp_path):
    remote = FakeQdrant()
    remote.offline = True
    index = make_index(tmp_path, remote)
    import_doc(index)
    assert not index.capabilities().semantic
    assert index.search(pot_id="p", query="ERR_QUOTA_EXCEEDED").hits
    assert "lexical" in index.status().detail
    report = index.drain()
    assert report.remaining == 5 and report.detail
    remote.offline = False
    assert index.drain().remaining == 0
    assert index.capabilities().semantic
    assert index.search(pot_id="p", query="quota").semantic_candidates == 3


def test_lost_upsert_ack_reuses_the_same_ids_and_cached_embeddings(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.transport.lose_upsert_ack = True
    first = index.drain()
    ids = set(index.transport.collections[index.collection])
    assert first.embedded == 5 and first.remaining == 5 and first.detail
    assert index.drain().remaining == 0
    assert set(index.transport.collections[index.collection]) == ids
    assert index.transport.upserts[-1] == index.transport.upserts[-2]
    assert index.embedder.calls == 1


def test_collection_recreation_and_lost_create_ack_republish_without_embedding(
    tmp_path,
):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    old_ids = set(index.transport.collections[index.collection])
    del index.transport.collections[index.collection]
    index.transport.lose_create_ack = True
    assert index.drain().remaining == 5
    index.close()
    resumed = make_index(tmp_path, index.transport, index.embedder)
    assert resumed.drain().remaining == 0
    assert set(resumed.transport.collections[resumed.collection]) == old_ids
    assert resumed.embedder.calls == 1


def test_replacement_reused_row_ids_cannot_match_old_vectors(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    before = set(index.transport.collections[index.collection])
    import_doc(index, text="replacement unrelated material")
    # Remote has all old points until the next drain, but SQLite invalidation
    # must make them ineligible immediately, even when row IDs are reused.
    result = index.search(pot_id="p", query="quota")
    assert result.semantic_candidates == 0
    assert index.pending_count() == 10
    assert index.drain().remaining == 0
    assert not before.intersection(index.transport.collections[index.collection])


def test_delete_lost_ack_is_durable_and_never_returns_removed_text(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    index.drop_document(pot_id="p", slug="guide")
    index.transport.lose_delete_ack = True
    assert index.drain().remaining == 5
    assert not index.search(pot_id="p", query="quota").hits
    assert index.drain().remaining == 0
    assert not index.transport.collections[index.collection]


def test_existing_sqlite_embeddings_migrate_without_reembedding(tmp_path):
    embedder = Embedder()
    old = SqliteFtsResourceIndex(home=tmp_path, embedder=embedder)
    import_doc(old)
    conn = old._connect()
    conn.execute(
        "UPDATE windows SET embedding=?,embedder=?,dimensions=?",
        (pack_vector([1, 0, 0]), embedder.name, 3),
    )
    conn.commit()
    old.close()
    index = make_index(tmp_path, embedder=embedder)
    report = index.drain()
    assert report.embedded == 0 and report.remaining == 0
    assert embedder.calls == 0
    assert len(index.transport.collections[index.collection]) == 5


def test_model_change_isolates_collections_and_old_process_cannot_write(tmp_path):
    first = make_index(tmp_path)
    import_doc(first)
    first.drain()
    old_collection = first.collection
    new_embedder = Embedder()
    new_embedder.name = "model-b"
    second = make_index(tmp_path, first.transport, new_embedder)
    second.status()
    assert second.collection != old_collection
    assert (
        first.drain().detail
        == "resource embedder changed; reopen the index; only lexical retrieval is available"
    )
    assert second.drain().remaining == 0
    assert not first.transport.collections[old_collection]
    assert second.search(pot_id="p", query="quota").semantic_candidates == 3


def test_embedding_snapshot_cannot_fill_a_replaced_window(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.embedder.on_embed = lambda: import_doc(index, text="new replacement content")
    assert index.drain().embedded == 0
    assert index.pending_count() == 5
    assert index.drain().remaining == 0


def test_filters_and_namespaces_isolate_pots_documents_and_databases(tmp_path):
    remote = FakeQdrant()
    index = make_index(tmp_path / "one", remote)
    for pot, doc in [("p", "first"), ("p", "second"), ("other", "third")]:
        import_doc(index, pot=pot, doc=doc)
    index.drain()
    result = index.search(pot_id="p", doc="first", query="quota")
    assert result.semantic_candidates == 3
    assert {h.doc for h in result.hits} == {"first"}
    assert remote.queries[-1][0] == {
        "must": [
            {"key": "namespace", "match": {"value": index._namespace}},
            {"key": "pot_id", "match": {"value": "p"}},
            {"key": "doc", "match": {"value": "first"}},
        ]
    }
    other = make_index(tmp_path / "two", remote)
    import_doc(other)
    other.drain()
    assert index.collection != other.collection
    assert all(
        set(p["payload"]) == {"namespace", "pot_id", "doc"}
        for points in remote.collections.values()
        for p in points.values()
    )


def test_stale_top_results_are_paged_until_valid_points_are_found(tmp_path):
    index = make_index(tmp_path)
    # 65 old documents force more than one remote page after replacement.
    for n in range(65):
        import_doc(index, doc=f"old-{n}")
    while index.pending_count():
        index.drain()
    for n in range(65):
        index.drop_document(pot_id="p", slug=f"old-{n}")
    import_doc(index, doc="current")
    # Fill the new points directly to retain the deliberately stale remote set.
    conn = index._connect()
    index._drain_batch(conn, pot_id=None, size=64)
    from potpie_context_engine.adapters.outbound.resources.index import (
        qdrant_sync as sync,
    )

    sync.prepare_points(
        conn,
        collection=index.collection,
        model="test-v1",
        dimensions=3,
        pot_id=None,
        size=64,
    )
    sync.publish(
        conn,
        index.transport,
        collection=index.collection,
        namespace=index._namespace,
        model="test-v1",
        dimensions=3,
        pot_id=None,
        size=64,
    )
    result = index.search(pot_id="p", query="quota")
    assert result.semantic_candidates == 3
    assert len(index.transport.queries) >= 2


def test_two_writers_share_point_identity_and_do_not_duplicate_publications(tmp_path):
    remote = FakeQdrant()
    first = make_index(tmp_path, remote)
    import_doc(first)
    second = make_index(tmp_path, remote)
    second.status()
    errors = []

    def run(index):
        try:
            index.drain()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors
    assert all(not t.is_alive() for t in threads)
    assert first.pending_count() == 0
    assert len(remote.collections[first.collection]) == 5
    assert first._namespace == second._namespace


def test_drain_budget_bounds_embedding_and_publication(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    report = index.drain(budget=2)
    assert report.embedded == 2 and report.remaining == 3
    assert len(index.transport.collections[index.collection]) == 2
    assert index.drain(budget=0).remaining == 3


def test_zero_and_invalid_vectors_do_not_publish_bad_cosine_points(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    conn = index._connect()
    conn.execute(
        "UPDATE windows SET embedding=?,embedder=?,dimensions=?",
        (pack_vector([0, 0, 0]), "test-v1", 3),
    )
    conn.commit()
    assert index.drain().remaining == 0
    assert not index.transport.collections[index.collection]
    conn.execute("UPDATE windows SET embedding=?", (pack_vector([float("nan"), 0, 0]),))
    conn.commit()
    report = index.drain()
    assert report.remaining and "invalid dimensions or values" in report.detail


def test_late_upsert_after_successful_delete_cannot_hide_live_results(tmp_path):
    index = make_index(tmp_path)
    for n in range(41):
        import_doc(index, doc=f"old-{n}", text="quota")
    index.drain()
    delayed = dict(index.transport.collections[index.collection])
    for n in range(41):
        index.drop_document(pot_id="p", slug=f"old-{n}")
    assert index.drain().remaining == 0
    import_doc(index, doc="current", text="current material")
    index.drain()
    # Model a remote request accepted before a timeout but applied after the
    # client's subsequent delete was already acknowledged.
    index.transport.collections[index.collection].update(delayed)
    result = index.search(pot_id="p", query="quota", limit=1)
    assert result.semantic_candidates == 3
    assert index.pending_count() == 205
    assert index.drain().remaining == 0
    assert len(index.transport.collections[index.collection]) == 5


def test_disabling_embedder_still_drains_deletion_obligations(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    remote, collection = index.transport, index.collection
    index.close()
    cleanup = QdrantHybridResourceIndex(home=tmp_path, transport=remote, embedder=None)
    assert cleanup.drain_when_unavailable
    cleanup.drop_document(pot_id="p", slug="guide")
    assert cleanup.pending_count() == 5
    assert cleanup.drain().remaining == 0
    assert not remote.collections[collection]
    result = cleanup.search(pot_id="p", query="quota")
    assert result.match_mode == "lexical"
    assert "no valid embedder" in result.detail


def test_fresh_index_without_embedder_needs_no_service_or_worker(tmp_path):
    remote = FakeQdrant()
    remote.offline = True
    index = QdrantHybridResourceIndex(home=tmp_path, transport=remote, embedder=None)
    assert not index.drain_when_unavailable
    assert index.drain().remaining == 0
    assert index.drain().detail is None
    assert not remote.collections


def test_failed_projection_initialization_is_retried_on_next_open(
    tmp_path, monkeypatch
):
    from potpie_context_engine.adapters.outbound.resources.index import qdrant_sync

    index = make_index(tmp_path)
    initialize = qdrant_sync.initialize
    attempted = []

    def fail_once(conn):
        attempted.append(conn)
        if len(attempted) == 1:
            raise sqlite3.OperationalError("database is locked")
        return initialize(conn)

    monkeypatch.setattr(qdrant_sync, "initialize", fail_once)
    assert index._try_connect() is None
    assert index._conn is None
    # The failed handle must also be released, so retries do not leak handles
    # or retain a transaction that blocks the next connection's initialization.
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        attempted[0].execute("SELECT 1")
    import_doc(index)
    assert len(attempted) == 2 and index._namespace
    assert index._open_error is None
    assert index.drain().remaining == 0
    assert index.search(pot_id="p", query="quota").semantic_candidates == 3
