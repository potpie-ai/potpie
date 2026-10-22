"""Durable SQLite bookkeeping for the Qdrant vector projection.

SQLite remains authoritative. Point identities survive retries; deleting a window
atomically records its remote tombstone. Network writes run under the same SQLite
writer transaction as their acknowledgements, preventing an older writer from
resurrecting a point after another process has removed it. Retired identities
remain recorded after deletion acknowledgement: a timed-out remote request can
finish late, so query validation must continue to recognize and reject it.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import uuid
from array import array
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .qdrant_http import QdrantError

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS qdrant_windows(
        window_id INTEGER PRIMARY KEY REFERENCES windows(id) ON DELETE CASCADE,
        token TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS qdrant_points(
        window_id INTEGER NOT NULL REFERENCES windows(id) ON DELETE CASCADE,
        collection TEXT NOT NULL, point_id TEXT NOT NULL UNIQUE,
        pot_id TEXT NOT NULL, embedding BLOB NOT NULL, synced INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(window_id, collection))""",
    """CREATE INDEX IF NOT EXISTS qdrant_pending
        ON qdrant_points(collection, pot_id) WHERE synced=0""",
    """CREATE TABLE IF NOT EXISTS qdrant_deletes(
        point_id TEXT PRIMARY KEY, collection TEXT NOT NULL, pot_id TEXT NOT NULL,
        completed INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TRIGGER IF NOT EXISTS qdrant_window_insert AFTER INSERT ON windows
        BEGIN INSERT INTO qdrant_windows(window_id,token)
        VALUES(NEW.id,lower(hex(randomblob(16)))); END""",
    """CREATE TRIGGER IF NOT EXISTS qdrant_point_delete AFTER DELETE ON qdrant_points
        BEGIN INSERT OR IGNORE INTO qdrant_deletes(point_id,collection,pot_id)
        VALUES(OLD.point_id,OLD.collection,OLD.pot_id); END""",
    """CREATE TRIGGER IF NOT EXISTS qdrant_vector_change
        AFTER UPDATE OF embedding,embedder,dimensions ON windows
        WHEN OLD.embedding IS NOT NEW.embedding OR OLD.embedder IS NOT NEW.embedder
            OR OLD.dimensions IS NOT NEW.dimensions
        BEGIN DELETE FROM qdrant_points WHERE window_id=OLD.id; END""",
)


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def initialize(conn: sqlite3.Connection) -> str:
    """Install triggers before model reconciliation can invalidate old vectors."""
    with write_transaction(conn):
        for sql in _SCHEMA:
            conn.execute(sql)
        if "completed" not in {
            row[1] for row in conn.execute("PRAGMA table_info(qdrant_deletes)")
        }:
            conn.execute(
                "ALTER TABLE qdrant_deletes ADD COLUMN completed INTEGER NOT NULL DEFAULT 0"
            )
        conn.execute(
            "INSERT OR IGNORE INTO qdrant_windows(window_id,token) "
            "SELECT id,lower(hex(randomblob(16))) FROM windows"
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES('qdrant_namespace',?)",
            (uuid.uuid4().hex,),
        )
        namespace = conn.execute(
            "SELECT value FROM meta WHERE key='qdrant_namespace'"
        ).fetchone()[0]
    return str(namespace)


def collection_name(namespace: str, model: str, dimensions: int) -> str:
    digest = hashlib.sha256(f"{model}\0{dimensions}".encode()).hexdigest()[:16]
    return f"potpie_resources_{namespace}_{digest}"


def pack_vector(values: Sequence[float]) -> bytes:
    """Keep the existing SQLite float32 representation for offline recovery."""
    return array("f", [float(value) for value in values]).tobytes()


def unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    vector = array("f")
    try:
        vector.frombytes(blob)
    except (ValueError, TypeError):
        raise QdrantError("stored embedding has an invalid float32 encoding") from None
    if len(vector) != dimensions or any(not math.isfinite(v) for v in vector):
        raise QdrantError("stored embedding has invalid dimensions or values")
    return list(vector)


def model_matches(conn: sqlite3.Connection, model: str, dimensions: int) -> bool:
    metadata = dict(
        conn.execute(
            "SELECT key,value FROM meta WHERE key IN ('embedder','dimensions')"
        )
    )
    return metadata == {"embedder": model, "dimensions": str(dimensions)}


def prepare_points(
    conn: sqlite3.Connection,
    *,
    collection: str,
    model: str,
    dimensions: int,
    pot_id: str | None,
    size: int,
) -> None:
    """Commit stable retry identities before a remote operation can start."""
    scope = " AND w.pot_id=?" if pot_id else ""
    params: list[Any] = [collection, model, dimensions]
    if pot_id:
        params.append(pot_id)
    params.append(size)
    with write_transaction(conn):
        if not model_matches(conn, model, dimensions):
            raise QdrantError("resource embedder changed; reopen the index")
        rows = conn.execute(
            "SELECT w.id,w.pot_id,w.embedding FROM windows w "
            "LEFT JOIN qdrant_points p ON p.window_id=w.id AND p.collection=? "
            "WHERE p.point_id IS NULL AND w.embedding IS NOT NULL "
            "AND w.embedder=? AND w.dimensions=?" + scope + " ORDER BY w.id LIMIT ?",
            params,
        ).fetchall()
        conn.executemany(
            "INSERT INTO qdrant_points(window_id,collection,point_id,pot_id,embedding) "
            "VALUES(?,?,?,?,?)",
            [
                (r["id"], collection, str(uuid.uuid4()), r["pot_id"], r["embedding"])
                for r in rows
            ],
        )


def publish(
    conn: sqlite3.Connection,
    transport: Any,
    *,
    collection: str,
    namespace: str,
    model: str,
    dimensions: int,
    pot_id: str | None,
    size: int,
) -> int:
    scope = " AND p.pot_id=?" if pot_id else ""
    params: list[Any] = [collection]
    if pot_id:
        params.append(pot_id)
    params.append(size)
    with write_transaction(conn):
        if not model_matches(conn, model, dimensions):
            raise QdrantError("resource embedder changed; reopen the index")
        rows = conn.execute(
            "SELECT p.point_id,p.embedding,w.pot_id,w.doc FROM qdrant_points p "
            "JOIN windows w ON w.id=p.window_id "
            "WHERE p.collection=? AND p.synced=0" + scope + " ORDER BY w.id LIMIT ?",
            params,
        ).fetchall()
        points = []
        for row in rows:
            vector = unpack_vector(row["embedding"], dimensions)
            # Cosine is undefined for a zero vector. Its durable synced state
            # records that it was intentionally omitted, avoiding endless work.
            if any(vector):
                points.append(
                    {
                        "id": row["point_id"],
                        "vector": vector,
                        "payload": {
                            "namespace": namespace,
                            "pot_id": row["pot_id"],
                            "doc": row["doc"],
                        },
                    }
                )
        transport.upsert(collection, points)
        conn.executemany(
            "UPDATE qdrant_points SET synced=1 WHERE point_id=?",
            [(r["point_id"],) for r in rows],
        )
    return len(rows)


def delete_pending(
    conn: sqlite3.Connection, transport: Any, *, pot_id: str | None, size: int
) -> int:
    scope = " AND pot_id=?" if pot_id else ""
    params: list[Any] = [pot_id] if pot_id else []
    params.append(size)
    with write_transaction(conn):
        rows = conn.execute(
            "SELECT * FROM qdrant_deletes WHERE completed=0"
            + scope
            + " ORDER BY rowid LIMIT ?",
            params,
        ).fetchall()
        collections: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            collections[row["collection"]].append(row["point_id"])
        for collection, point_ids in collections.items():
            transport.delete(collection, point_ids)
            conn.executemany(
                "UPDATE qdrant_deletes SET completed=1 WHERE point_id=?",
                [(point_id,) for point_id in point_ids],
            )
    return len(rows)


def pending_count(
    conn: sqlite3.Connection, *, collection: str, pot_id: str | None
) -> int:
    scope = " AND w.pot_id=?" if pot_id else ""
    params: list[Any] = [collection]
    if pot_id:
        params.append(pot_id)
    windows = conn.execute(
        "SELECT COUNT(*) FROM windows w LEFT JOIN qdrant_points p "
        "ON p.window_id=w.id AND p.collection=? "
        "WHERE (w.embedding IS NULL OR p.point_id IS NULL OR p.synced=0)" + scope,
        params,
    ).fetchone()[0]
    deletes = conn.execute(
        "SELECT COUNT(*) FROM qdrant_deletes WHERE completed=0"
        + (" AND pot_id=?" if pot_id else ""),
        (pot_id,) if pot_id else (),
    ).fetchone()[0]
    return int(windows) + int(deletes)
