"""SQLite BM25 and durable embedding cache, with exact cosine search in Qdrant.

The service receives vectors, an index namespace, and pot/document identifiers. SQLite owns
source text, point lifetimes and the retry log, so an unavailable service leaves
lexical retrieval usable and publication resumes without re-embedding.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from potpie_context_engine.core.ports.resource_index import (
    DEFAULT_DRAIN_BUDGET,
    DrainReport,
    IndexCapabilities,
    ResourceIndexStatus,
)
from . import qdrant_sync as sync
from .qdrant_http import QdrantError, QdrantHttp
from .sqlite_fts import (
    ArmMatch,
    SqliteFtsResourceIndex,
    WINDOW_KIND_SECTION,
    _clean_snippet,
    iter_window_texts,
)

EMBED_BATCH = 64
pack_vector = sync.pack_vector
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _EmbeddingBatch:
    embedded: int = 0
    skipped: int = 0
    attempted: int = 0
    last_window_id: int = 0
    failed: bool = False


@dataclass(slots=True)
class QdrantHybridResourceIndex(SqliteFtsResourceIndex):
    profile: str = "qdrant_hybrid"
    qdrant_url: str = field(
        default_factory=lambda: os.getenv(
            "CONTEXT_ENGINE_QDRANT_URL", "http://127.0.0.1:6333"
        )
    )
    qdrant_api_key: str | None = field(
        default_factory=lambda: os.getenv("CONTEXT_ENGINE_QDRANT_API_KEY"), repr=False
    )
    transport: Any = field(default=None, repr=False)
    _namespace: str = field(default="", init=False, repr=False)
    _health_error: str | None = field(
        default="Qdrant has not been checked", init=False, repr=False
    )
    _health_checked: float = field(default=float("-inf"), init=False, repr=False)
    _owns_transport: bool = field(default=False, init=False, repr=False)
    _stale_point_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def _reconcile_embedder(self, conn: sqlite3.Connection) -> None:
        self._namespace = sync.initialize(conn)
        with self._lock:
            # The base reconciler commits. Acquire the writer lock first so two
            # processes cannot make decisions from different metadata snapshots.
            conn.execute("BEGIN IMMEDIATE")
            try:
                name, dimensions = self._embedder_identity()
                if name is not None and not sync.model_matches(conn, name, dimensions):
                    conn.execute("UPDATE qdrant_windows SET skipped_collection=NULL")
                SqliteFtsResourceIndex._reconcile_embedder(self, conn)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    @property
    def drain_when_unavailable(self) -> bool:
        if self.embedder is not None:
            return True
        conn = self._try_connect()
        if conn is None:
            return False
        with self._lock:
            return bool(
                conn.execute(
                    "SELECT EXISTS(SELECT 1 FROM qdrant_points) OR "
                    "EXISTS(SELECT 1 FROM qdrant_deletes WHERE completed=0)"
                ).fetchone()[0]
            )

    @property
    def collection(self) -> str:
        name, dimensions = self._embedder_identity()
        return sync.collection_name(self._namespace, name or "none", dimensions or 0)

    def _client(self) -> Any:
        if self.transport is None:
            self.transport = QdrantHttp(self.qdrant_url, self.qdrant_api_key)
            self._owns_transport = True
        return self.transport

    def _failed(self, error: QdrantError) -> str:
        self._health_error = f"{error}; only lexical retrieval is available"
        self._health_checked = time.monotonic()
        return self._health_error

    def _vector_error(self, *, refresh: bool = False) -> str | None:
        name, dimensions = self._embedder_identity()
        if name is None or dimensions is None or dimensions < 1:
            self._health_error = (
                "no valid embedder is wired; only lexical retrieval is available"
            )
            return self._health_error
        conn = self._try_connect()
        if conn is None:
            return self._open_error or "index database unavailable"
        with self._lock:
            if not sync.model_matches(conn, name, dimensions):
                return self._failed(
                    QdrantError("resource embedder changed; reopen the index")
                )
            # Search already holds a SQLite snapshot; never open a nested write
            # transaction or repeat a health probe inside that snapshot.
            age = time.monotonic() - self._health_checked
            if conn.in_transaction or (
                not refresh and age < (1 if self._health_error else 2)
            ):
                return self._health_error
            try:

                def before_create() -> None:
                    # Commit before remote creation: a lost response or process
                    # crash must leave a durable republish obligation.
                    with sync.write_transaction(conn):
                        conn.execute(
                            "UPDATE qdrant_points SET synced=0 "
                            "WHERE collection=? AND synced=1",
                            (self.collection,),
                        )

                self._client().ensure_collection(
                    self.collection, dimensions, on_missing=before_create
                )
                self._health_error = None
                self._health_checked = time.monotonic()
            except (QdrantError, sqlite3.Error) as exc:
                return self._failed(QdrantError(str(exc)))
            return self._health_error

    def capabilities(self) -> IndexCapabilities:
        vectors = self._vector_error() is None
        return IndexCapabilities(
            profile=self.profile,
            lexical=True,
            semantic=vectors,
            hybrid=vectors,
            snippets=True,
            incremental=True,
        )

    def search(self, **kwargs: Any) -> Any:
        # A deferred read snapshot keeps lexical results and hydration consistent
        # while allowing other connections to publish or import during paging.
        self._vector_error()
        conn = self._try_connect()
        if conn is None:
            return SqliteFtsResourceIndex.search(self, **kwargs)
        with self._lock:
            with sync.read_transaction(conn):
                result = SqliteFtsResourceIndex.search(self, **kwargs)
            self._rearm_stale_points(conn)
            if self._health_error:
                result = dataclasses.replace(
                    result, detail=self._health_error, match_mode="lexical"
                )
            return result

    def _rearm_stale_points(self, conn: sqlite3.Connection) -> None:
        if not self._stale_point_ids:
            return
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        try:
            # Cleanup must not make a read wait behind a remote publication.
            # Keep the ids for another search if a writer currently owns SQLite.
            conn.execute("PRAGMA busy_timeout=0")
            with sync.write_transaction(conn):
                conn.executemany(
                    "UPDATE qdrant_deletes SET completed=0 WHERE point_id=?",
                    [(point_id,) for point_id in self._stale_point_ids],
                )
            self._stale_point_ids.clear()
        except sqlite3.Error as exc:
            logger.debug("resource stale-point cleanup deferred: %s", exc)
        finally:
            conn.execute(f"PRAGMA busy_timeout={int(timeout)}")

    def _semantic_arm(
        self,
        conn: sqlite3.Connection,
        *,
        pot_id: str,
        query: str,
        limit: int,
        doc: str | None,
    ) -> dict[int, ArmMatch]:
        if not (query or "").strip() or self.embedder is None:
            return {}
        _, dimensions = self._embedder_identity()
        try:
            vector = list(self.embedder.embed(query))
            if (
                len(vector) != dimensions
                or not all(math.isfinite(v) for v in vector)
                or not any(vector)
            ):
                return {}
        except Exception:  # A model error must not prevent lexical retrieval.
            self._failed(QdrantError("resource query embedding failed"))
            return {}
        must = [
            {"key": "namespace", "match": {"value": self._namespace}},
            {"key": "pot_id", "match": {"value": pot_id}},
        ]
        if doc:
            must.append({"key": "doc", "match": {"value": doc}})
        # Every point we ever sent remains represented by current state or a
        # retained tombstone, including acknowledged deletes. This gives pagination a
        # finite upper bound without letting stale top hits hide live matches.
        max_points = conn.execute(
            "SELECT (SELECT COUNT(*) FROM qdrant_points WHERE collection=? AND pot_id=?)"
            "+(SELECT COUNT(*) FROM qdrant_deletes WHERE collection=? AND pot_id=?)",
            (self.collection, pot_id, self.collection, pot_id),
        ).fetchone()[0]
        best: dict[int, ArmMatch] = {}
        page_size = max(64, limit * 4)
        offset = 0
        try:
            while offset < max_points and len(best) < limit:
                hits = self._client().query(
                    self.collection, vector, {"must": must}, page_size, offset
                )
                if not hits:
                    break
                for hit in hits:
                    row = conn.execute(
                        "SELECT w.*,c.text AS body FROM qdrant_points p "
                        "JOIN windows w ON w.id=p.window_id JOIN chunks c ON c.id=w.chunk_id "
                        "WHERE p.point_id=? AND p.collection=? AND p.synced=1 "
                        "AND p.embedding=w.embedding AND w.pot_id=? "
                        "AND w.embedder=? AND w.dimensions=?",
                        (
                            str(hit["id"]),
                            self.collection,
                            pot_id,
                            *self._embedder_identity(),
                        ),
                    ).fetchone()
                    if row is None:
                        # A timed-out upsert may complete after a deletion ACK.
                        # Retained identities keep paging complete and let a
                        # later sighting durably request another delete.
                        self._stale_point_ids.add(str(hit["id"]))
                        continue
                    if doc and row["doc"] != doc:
                        continue
                    chunk_id = int(row["chunk_id"])
                    if chunk_id in best:
                        continue
                    text = (
                        str(row["text"] or "")
                        if row["kind"] == WINDOW_KIND_SECTION
                        else str(row["body"])[
                            int(row["start"]) : int(row["start"]) + int(row["length"])
                        ]
                    )
                    best[chunk_id] = ArmMatch(
                        snippet=_clean_snippet(text),
                        similarity=max(0.0, min(1.0, float(hit["score"]))),
                    )
                    if len(best) >= limit:
                        break
                offset += len(hits)
                if len(hits) < page_size:
                    break
        except QdrantError as exc:
            self._failed(exc)
            return {}
        return best

    def _drain_batch(
        self,
        conn: sqlite3.Connection,
        *,
        pot_id: str | None,
        size: int,
        after_id: int = 0,
    ) -> _EmbeddingBatch:
        scope = " AND w.pot_id=?" if pot_id else ""
        params: list[Any] = [self.collection, after_id]
        if pot_id:
            params.append(pot_id)
        params.append(size)
        with self._lock:
            rows = conn.execute(
                "SELECT w.*,i.token,c.text AS body FROM windows w "
                "JOIN qdrant_windows i ON i.window_id=w.id JOIN chunks c ON c.id=w.chunk_id "
                "WHERE w.embedding IS NULL AND i.skipped_collection IS NOT ? AND w.id>?"
                + scope
                + " ORDER BY w.id LIMIT ?",
                params,
            ).fetchall()
        if not rows:
            return _EmbeddingBatch()
        texts = list(
            iter_window_texts(
                rows, bodies={int(r["chunk_id"]): str(r["body"]) for r in rows}
            )
        )
        name, dimensions = self._embedder_identity()

        def validate(vector: Any) -> bytes:
            blob = pack_vector(vector)
            if not any(sync.unpack_vector(blob, dimensions)):
                raise ValueError("zero embedding has no cosine similarity")
            return blob

        try:
            vectors = list(self.embedder.embed_many(texts))
            if len(vectors) != len(rows):
                raise ValueError("incomplete embedding batch")
        except Exception:
            vectors = [None] * len(rows)
        completed: list[tuple[sqlite3.Row, bytes | None]] = []
        failed = False
        for row, text, vector in zip(rows, texts, vectors, strict=True):
            try:
                blob = validate(vector)
            except (TypeError, ValueError, OverflowError, QdrantError):
                # A partial/malformed batch can be a batching defect. Retry that
                # input alone before deciding that its vector is unusable.
                try:
                    vector = self.embedder.embed(text)
                except Exception:
                    # Raised model errors may be transient: leave this row
                    # pending, but retain successful siblings from this batch.
                    failed = True
                    continue
                try:
                    blob = validate(vector)
                except (TypeError, ValueError, OverflowError, QdrantError):
                    blob = None
            completed.append((row, blob))
        updated = skipped = 0
        with self._lock, sync.write_transaction(conn):
            if not sync.model_matches(conn, name, dimensions):
                raise QdrantError("resource embedder changed; reopen the index")
            for row, blob in completed:
                if blob is None:
                    skipped += conn.execute(
                        "UPDATE qdrant_windows SET skipped_collection=? "
                        "WHERE window_id=? AND token=? AND EXISTS(SELECT 1 FROM windows "
                        "WHERE id=qdrant_windows.window_id AND embedding IS NULL)",
                        (self.collection, row["id"], row["token"]),
                    ).rowcount
                    continue
                updated += conn.execute(
                    "UPDATE windows SET embedding=?,embedder=?,dimensions=? WHERE id=? "
                    "AND embedding IS NULL AND EXISTS(SELECT 1 FROM qdrant_windows "
                    "WHERE window_id=windows.id AND token=?)",
                    (blob, name, dimensions, row["id"], row["token"]),
                ).rowcount
        return _EmbeddingBatch(
            embedded=updated,
            skipped=skipped,
            attempted=len(rows),
            last_window_id=int(rows[-1]["id"]),
            failed=failed,
        )

    def drain(
        self, *, pot_id: str | None = None, budget: int = DEFAULT_DRAIN_BUDGET
    ) -> DrainReport:
        started = time.monotonic()
        embedded = batches = processed = 0
        conn = self._try_connect()
        if self.embedder is None:
            # Removing a document must remove its remote projection even after
            # embedding has been disabled. No health probe/collection creation is
            # necessary: each durable tombstone already names its collection.
            error = None
            if conn is not None and budget > 0:
                try:
                    with self._lock:
                        processed += sync.delete_pending(
                            conn, self._client(), pot_id=pot_id, size=budget
                        )
                except QdrantError as exc:
                    error = self._failed(exc)
            return DrainReport(
                profile=self.profile,
                remaining=self.pending_count(pot_id=pot_id),
                elapsed_ms=int((time.monotonic() - started) * 1000),
                detail=error,
                processed=processed,
            )
        error = self._vector_error(refresh=True)
        if error is None and conn is not None and budget > 0:
            name, dimensions = self._embedder_identity()
            try:
                # Each category has a finite budget, so deletion/migration cannot
                # starve embedding or make a pass run indefinitely.
                with self._lock:
                    processed += sync.delete_pending(
                        conn, self._client(), pot_id=pot_id, size=budget
                    )
                attempted = after_id = 0
                while attempted < budget:
                    batch = self._drain_batch(
                        conn,
                        pot_id=pot_id,
                        size=min(EMBED_BATCH, budget - attempted),
                        after_id=after_id,
                    )
                    # Visit each row at most once in this pass. Retryable model
                    # failures must not hide later rows or consume an unbounded
                    # number of attempts within a finite drain budget.
                    attempted += batch.attempted
                    after_id = batch.last_window_id
                    embedded += batch.embedded
                    processed += batch.embedded + batch.skipped
                    if batch.attempted:
                        batches += 1
                    if batch.failed:
                        error = self._failed(
                            QdrantError("resource batch embedding failed")
                        )
                    if not batch.attempted:
                        break
                published = 0
                while published < budget:
                    size = min(EMBED_BATCH, budget - published)
                    with self._lock:
                        sync.prepare_points(
                            conn,
                            collection=self.collection,
                            model=name,
                            dimensions=dimensions,
                            pot_id=pot_id,
                            size=size,
                        )
                        count = sync.publish(
                            conn,
                            self._client(),
                            collection=self.collection,
                            namespace=self._namespace,
                            model=name,
                            dimensions=dimensions,
                            pot_id=pot_id,
                            size=size,
                        )
                    if not count:
                        break
                    published += count
                    processed += count
            except QdrantError as exc:
                error = self._failed(exc)
        return DrainReport(
            profile=self.profile,
            embedded=embedded,
            remaining=self.pending_count(pot_id=pot_id),
            batches=batches,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            detail=error,
            processed=processed,
        )

    def index_document(self, **kwargs: Any) -> Any:
        report = SqliteFtsResourceIndex.index_document(self, **kwargs)
        if self.embedder is not None:
            return dataclasses.replace(report, pending_embeddings=report.windows)
        return report

    def pending_count(self, *, pot_id: str | None = None) -> int:
        conn = self._try_connect()
        if conn is None:
            return 0
        with self._lock:
            if self.embedder is None:
                return int(
                    conn.execute(
                        "SELECT COUNT(*) FROM qdrant_deletes WHERE completed=0"
                        + (" AND pot_id=?" if pot_id else ""),
                        (pot_id,) if pot_id else (),
                    ).fetchone()[0]
                )
            return sync.pending_count(conn, collection=self.collection, pot_id=pot_id)

    def status(self, *, pot_id: str | None = None) -> ResourceIndexStatus:
        report = SqliteFtsResourceIndex.status(self, pot_id=pot_id)
        reason = self._vector_error()
        if reason and report.ready:
            return dataclasses.replace(
                report, detail="; ".join(p for p in (reason, report.detail) if p)
            )
        return report

    def close(self) -> None:
        with self._lock:
            SqliteFtsResourceIndex.close(self)
            if self._owns_transport and self.transport is not None:
                self.transport.close()
                self.transport = None
                self._owns_transport = False
            self._health_checked = float("-inf")


__all__ = ["EMBED_BATCH", "QdrantHybridResourceIndex", "pack_vector"]
