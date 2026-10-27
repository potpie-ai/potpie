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
from potpie_context_engine.application.services.resource_facade import ResourceFacade
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
        manifest = dataclasses.replace(
            manifest,
            revision=2,
            sections=tuple(
                dataclasses.replace(section, summary=text)
                for section in manifest.sections
            ),
        )
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


def test_wait_republishes_all_cached_vectors_after_collection_loss(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    del index.transport.collections[index.collection]
    facade = ResourceFacade(store=None, index=index)

    report = facade.index_build(pot_id="p", budget=2, wait=True)

    assert report.remaining == 0
    assert report.embedded == 0
    assert len(index.transport.collections[index.collection]) == 5
    assert index.embedder.calls == 1


@pytest.mark.parametrize("embeddings_enabled", [False, True])
def test_wait_deletes_all_tombstones_across_multiple_budgets(
    tmp_path, embeddings_enabled
):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    collection = index.collection
    index.drop_document(pot_id="p", slug="guide")
    if not embeddings_enabled:
        index.embedder = None
    facade = ResourceFacade(store=None, index=index)

    report = facade.index_build(pot_id="p", budget=2, wait=True)

    assert report.remaining == 0
    assert report.embedded == 0
    assert not index.transport.collections[collection]


def test_wait_returns_when_qdrant_cannot_make_progress(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.transport.offline = True
    facade = ResourceFacade(store=None, index=index)

    report = facade.index_build(pot_id="p", budget=2, wait=True)

    assert report.remaining == 5
    assert report.detail
    assert report.processed == 0
    assert not index.transport.upserts


def test_wait_finishes_terminal_skips_across_multiple_budgets(tmp_path):
    class InvalidEmbedder(Embedder):
        def embed(self, text):
            return [0, 0, 0]

    index = make_index(tmp_path, embedder=InvalidEmbedder())
    import_doc(index)
    facade = ResourceFacade(store=None, index=index)

    report = facade.index_build(pot_id="p", budget=2, wait=True)

    assert report.remaining == 0
    assert report.embedded == 0
    assert index.embedder.calls == 3
    assert not index.transport.upserts
    assert index.search(pot_id="p", query="ERR_QUOTA_EXCEEDED").hits


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
    assert report.remaining == 0 and report.detail is None
    assert not index.transport.collections[index.collection]
    assert (
        conn.execute("SELECT COUNT(*) FROM qdrant_points WHERE synced=2").fetchone()[0]
        == 5
    )


def test_search_keeps_a_read_snapshot_while_another_connection_replaces_document(
    tmp_path, monkeypatch
):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    writer = make_index(tmp_path, index.transport)
    writer._connect().execute("PRAGMA busy_timeout=50")
    query = index.transport.query

    def replace_during_query(*args):
        hits = query(*args)
        import_doc(writer, text="new unrelated replacement")
        return hits

    monkeypatch.setattr(index.transport, "query", replace_during_query)
    result = index.search(pot_id="p", query="quota")
    assert result.semantic_candidates == 3
    assert {hit.revision for hit in result.hits} == {1}
    assert not index._connect().in_transaction
    assert (
        writer._connect().execute("SELECT MIN(revision) FROM chunks").fetchone()[0] == 2
    )


def test_search_reads_while_another_connection_holds_writer_transaction(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    index._connect().execute("PRAGMA busy_timeout=50")
    writer = make_index(tmp_path, index.transport)
    conn = writer._connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = index.search(pot_id="p", query="quota")
    finally:
        conn.rollback()
    assert result.hits and result.semantic_candidates == 3
    assert result.detail is None


def test_busy_stale_cleanup_preserves_search_and_retries_after_snapshot(tmp_path):
    index = make_index(tmp_path)
    import_doc(index, doc="old")
    index.drain()
    delayed = dict(index.transport.collections[index.collection])
    index.drop_document(pot_id="p", slug="old")
    index.drain()
    import_doc(index, doc="current")
    index.drain()
    index.transport.collections[index.collection].update(delayed)
    writer = make_index(tmp_path, index.transport)
    conn = writer._connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = index.search(pot_id="p", query="quota")
        assert result.semantic_candidates == 3
        assert {hit.doc for hit in result.hits} == {"current"}
        assert index.pending_count() == 0
        assert index._stale_point_ids == set(delayed)
    finally:
        conn.rollback()
    assert index._connect().execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    assert index.search(pot_id="p", query="quota").semantic_candidates == 3
    assert index.pending_count() == 5
    assert not index._stale_point_ids
    assert index.drain().remaining == 0


@pytest.mark.parametrize("failure", ["raises", "short", "long", "malformed"])
def test_bad_batch_falls_back_to_individual_embeddings(tmp_path, failure):
    class BrokenBatch(Embedder):
        def embed_many(self, texts):
            self.calls += 1
            vectors = [self.embed(text) for text in texts]
            if failure == "raises":
                raise RuntimeError("batch unavailable")
            if failure == "short":
                return vectors[:-1]
            if failure == "long":
                return vectors + [vectors[0]]
            return [None, *vectors[1:]]

    index = make_index(tmp_path, embedder=BrokenBatch())
    import_doc(index)
    report = index.drain()
    assert report.embedded == 5 and report.remaining == 0 and report.detail is None
    assert len(index.transport.collections[index.collection]) == 5


@pytest.mark.parametrize(
    "invalid", [None, [1], [float("nan"), 0, 0], [float("inf"), 0, 0], [0, 0, 0]]
)
def test_invalid_individual_vectors_are_skipped_durably_without_blocking_good_rows(
    tmp_path, invalid
):
    class InvalidInput(Embedder):
        def embed(self, text):
            return invalid if "unembeddable" in text else super().embed(text)

    embedder = InvalidInput()
    index = make_index(tmp_path, embedder=embedder)
    import_doc(index, doc="bad", text="unembeddable")
    import_doc(index, doc="good")
    report = index.drain()
    assert report.embedded == 5 and report.remaining == 0 and report.detail is None
    assert len(index.transport.collections[index.collection]) == 5
    assert (
        index._connect()
        .execute(
            "SELECT COUNT(*) FROM qdrant_windows WHERE skipped_collection=?",
            (index.collection,),
        )
        .fetchone()[0]
        == 5
    )
    remote = index.transport
    index.close()
    resumed = make_index(tmp_path, remote, embedder)
    assert resumed.drain().remaining == 0
    assert embedder.calls == 1
    assert resumed.search(pot_id="p", query="unembeddable", doc="bad").hits
    import_doc(resumed, doc="bad", text="valid replacement")
    report = resumed.drain()
    assert report.embedded == 5 and report.remaining == 0
    assert len(remote.collections[resumed.collection]) == 10


def test_skipped_inputs_retry_after_model_change_even_when_switching_back(tmp_path):
    class InvalidEmbedder(Embedder):
        def embed(self, text):
            return [float("nan"), 0, 0]

    index = make_index(tmp_path, embedder=InvalidEmbedder())
    import_doc(index)
    assert index.drain().remaining == 0
    remote = index.transport
    index.close()
    different = Embedder()
    different.name = "model-b"
    switched = make_index(tmp_path, remote, different)
    assert switched.pending_count() == 5
    switched.close()
    original = make_index(tmp_path, remote)
    assert original.pending_count() == 5
    report = original.drain()
    assert report.embedded == 5 and report.remaining == 0


def test_retryable_failures_do_not_hide_later_batches_or_discard_successes(tmp_path):
    class TransientFailure(Embedder):
        failing = True

        def embed(self, text):
            if self.failing and "unembeddable" in text:
                raise RuntimeError("model temporarily unavailable for this input")
            return super().embed(text)

    embedder = TransientFailure()
    index = make_index(tmp_path, embedder=embedder)
    # A whole leading batch fails, followed by another bad row and valid rows.
    for number in range(13):
        import_doc(index, doc=f"bad-{number}", text="unembeddable")
    import_doc(index, doc="good")
    report = index.drain(budget=70)
    assert report.embedded == 5 and report.remaining == 65
    assert report.batches == 2 and "embedding failed" in report.detail
    assert len(index.transport.collections[index.collection]) == 5
    remote = index.transport
    index.close()
    embedder.failing = False
    resumed = make_index(tmp_path, remote, embedder)
    report = resumed.drain(budget=70)
    assert report.embedded == 65 and report.remaining == 0 and report.detail is None
    assert len(remote.collections[resumed.collection]) == 70


def test_skipped_embeddings_consume_the_drain_attempt_budget(tmp_path):
    class InvalidEmbedder(Embedder):
        def embed(self, text):
            return [float("nan"), 0, 0]

    index = make_index(tmp_path, embedder=InvalidEmbedder())
    import_doc(index)
    report = index.drain(budget=2)
    assert report.embedded == 0 and report.remaining == 3 and report.batches == 1
    assert index.drain(budget=3).remaining == 0


@pytest.mark.parametrize(
    "blob",
    [b"x", pack_vector([1]), pack_vector([float("nan"), 0, 0]), pack_vector([0, 0, 0])],
)
def test_corrupt_cached_vector_does_not_block_publication_and_can_be_repaired(
    tmp_path, blob
):
    index = make_index(tmp_path)
    import_doc(index)
    conn = index._connect()
    conn.execute(
        "UPDATE windows SET embedding=?,embedder=?,dimensions=?",
        (pack_vector([1, 0, 0]), "test-v1", 3),
    )
    window_id = conn.execute("SELECT MIN(id) FROM windows").fetchone()[0]
    conn.execute("UPDATE windows SET embedding=? WHERE id=?", (blob, window_id))
    conn.commit()
    report = index.drain()
    assert report.embedded == 0 and report.remaining == 0 and report.detail is None
    assert len(index.transport.collections[index.collection]) == 4
    assert (
        conn.execute(
            "SELECT synced FROM qdrant_points WHERE window_id=?", (window_id,)
        ).fetchone()[0]
        == 2
    )
    # Recreating the collection must keep terminal skipped points out of work.
    del index.transport.collections[index.collection]
    assert index.drain().remaining == 0
    assert len(index.transport.collections[index.collection]) == 4
    assert (
        conn.execute(
            "SELECT synced FROM qdrant_points WHERE window_id=?", (window_id,)
        ).fetchone()[0]
        == 2
    )
    conn.execute(
        "UPDATE windows SET embedding=? WHERE id=?",
        (pack_vector([1, 0, 0]), window_id),
    )
    conn.commit()
    assert index.drain().remaining == 0
    assert len(index.transport.collections[index.collection]) == 5
    assert index.embedder.calls == 0


def test_existing_projection_schema_migrates_skip_state_and_keeps_triggers(tmp_path):
    index = make_index(tmp_path)
    import_doc(index)
    index.drain()
    conn = index._connect()
    # Model the schema shipped before durable per-window skip state existed.
    conn.execute("DROP TRIGGER qdrant_skipped_vector_change")
    conn.execute("ALTER TABLE qdrant_windows DROP COLUMN skipped_collection")
    conn.commit()
    remote = index.transport
    index.close()
    resumed = make_index(tmp_path, remote)
    assert resumed.pending_count() == 0
    assert "skipped_collection" in {
        row[1]
        for row in resumed._connect().execute("PRAGMA table_info(qdrant_windows)")
    }
    import_doc(resumed, text="replacement")
    assert resumed.drain().remaining == 0
    resumed.drop_document(pot_id="p", slug="guide")
    assert resumed.drain().remaining == 0
    assert not remote.collections[resumed.collection]


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
