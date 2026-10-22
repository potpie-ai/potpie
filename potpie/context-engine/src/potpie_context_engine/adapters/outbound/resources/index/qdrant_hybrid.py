"""SQLite BM25 and durable embedding cache, with exact cosine search in Qdrant.

The service receives vectors, an index namespace, and pot/document identifiers. SQLite owns
source text, point lifetimes and the retry log, so an unavailable service leaves
lexical retrieval usable and publication resumes without re-embedding.
"""

from __future__ import annotations

import dataclasses
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

    def _reconcile_embedder(self, conn: sqlite3.Connection) -> None:
        self._namespace = sync.initialize(conn)
        with self._lock:
            # The base reconciler commits. Acquire the writer lock first so two
            # processes cannot make decisions from different metadata snapshots.
            conn.execute("BEGIN IMMEDIATE")
            try:
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
                            "UPDATE qdrant_points SET synced=0 WHERE collection=?",
                            (self.collection,),
                        )

                self._client().ensure_collection(
                    self.collection, dimensions, on_missing=before_create
                )
                self._health_error = None
                self._health_checked = time.monotonic()
            except QdrantError as exc:
                return self._failed(exc)
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
        # Serialize projection writers while paging, keeping their remote result
        # ordering and our SQLite hydration in one consistent document snapshot.
        self._vector_error()
        conn = self._try_connect()
        if conn is None:
            return SqliteFtsResourceIndex.search(self, **kwargs)
        with self._lock, sync.write_transaction(conn):
            result = SqliteFtsResourceIndex.search(self, **kwargs)
            if self._health_error:
                result = dataclasses.replace(
                    result, detail=self._health_error, match_mode="lexical"
                )
            return result

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
                        conn.execute(
                            "UPDATE qdrant_deletes SET completed=0 WHERE point_id=?",
                            (str(hit["id"]),),
                        )
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
        self, conn: sqlite3.Connection, *, pot_id: str | None, size: int
    ) -> int:
        scope = " AND w.pot_id=?" if pot_id else ""
        params: list[Any] = [pot_id] if pot_id else []
        params.append(size)
        with self._lock:
            rows = conn.execute(
                "SELECT w.*,i.token,c.text AS body FROM windows w "
                "JOIN qdrant_windows i ON i.window_id=w.id JOIN chunks c ON c.id=w.chunk_id "
                "WHERE w.embedding IS NULL" + scope + " ORDER BY w.id LIMIT ?",
                params,
            ).fetchall()
        if not rows:
            return 0
        texts = iter_window_texts(
            rows, bodies={int(r["chunk_id"]): str(r["body"]) for r in rows}
        )
        name, dimensions = self._embedder_identity()
        try:
            vectors = list(self.embedder.embed_many(texts))
            if len(vectors) != len(rows):
                raise ValueError("incomplete embedding batch")
            blobs = [pack_vector(v) for v in vectors]
            for blob in blobs:
                sync.unpack_vector(blob, dimensions)
        except Exception:
            raise QdrantError("resource batch embedding failed") from None
        updated = 0
        with self._lock, sync.write_transaction(conn):
            if not sync.model_matches(conn, name, dimensions):
                raise QdrantError("resource embedder changed; reopen the index")
            for row, blob in zip(rows, blobs, strict=True):
                updated += conn.execute(
                    "UPDATE windows SET embedding=?,embedder=?,dimensions=? WHERE id=? "
                    "AND embedding IS NULL AND EXISTS(SELECT 1 FROM qdrant_windows "
                    "WHERE window_id=windows.id AND token=?)",
                    (blob, name, dimensions, row["id"], row["token"]),
                ).rowcount
        return updated

    def drain(
        self, *, pot_id: str | None = None, budget: int = DEFAULT_DRAIN_BUDGET
    ) -> DrainReport:
        started = time.monotonic()
        embedded = batches = 0
        conn = self._try_connect()
        if self.embedder is None:
            # Removing a document must remove its remote projection even after
            # embedding has been disabled. No health probe/collection creation is
            # necessary: each durable tombstone already names its collection.
            error = None
            if conn is not None and budget > 0:
                try:
                    with self._lock:
                        sync.delete_pending(
                            conn, self._client(), pot_id=pot_id, size=budget
                        )
                except QdrantError as exc:
                    error = self._failed(exc)
            return DrainReport(
                profile=self.profile,
                remaining=self.pending_count(pot_id=pot_id),
                elapsed_ms=int((time.monotonic() - started) * 1000),
                detail=error,
            )
        error = self._vector_error(refresh=True)
        if error is None and conn is not None and budget > 0:
            name, dimensions = self._embedder_identity()
            try:
                # Each category has a finite budget, so deletion/migration cannot
                # starve embedding or make a pass run indefinitely.
                with self._lock:
                    sync.delete_pending(
                        conn, self._client(), pot_id=pot_id, size=budget
                    )
                while embedded < budget:
                    filled = self._drain_batch(
                        conn, pot_id=pot_id, size=min(EMBED_BATCH, budget - embedded)
                    )
                    if not filled:
                        break
                    embedded += filled
                    batches += 1
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
            except QdrantError as exc:
                error = self._failed(exc)
        return DrainReport(
            profile=self.profile,
            embedded=embedded,
            remaining=self.pending_count(pot_id=pot_id),
            batches=batches,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            detail=error,
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
