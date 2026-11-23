"""Local SQLite review queue for document-derived semantic claims."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.local_paths import default_home


def _ttl_days() -> int:
    try:
        return max(0, int(os.environ.get("POTPIE_FACT_INBOX_TTL_DAYS", "30")))
    except ValueError:
        return 30


@dataclass(slots=True)
class FactInbox:
    home: Path = field(default_factory=default_home)

    @property
    def path(self) -> Path:
        return self.home / "fact_inbox.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.home.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS fact_candidates (
                id TEXT PRIMARY KEY, pot_id TEXT NOT NULL, doc TEXT NOT NULL,
                revision INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, payload_json TEXT NOT NULL,
                reason TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                UNIQUE(pot_id, doc, revision, fingerprint)
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS fact_candidates_queue "
            "ON fact_candidates(pot_id, status, created_at)"
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS fact_extraction_chunks (
                pot_id TEXT NOT NULL, doc TEXT NOT NULL, revision INTEGER NOT NULL,
                chunk_uri TEXT NOT NULL, model_signature TEXT NOT NULL,
                text_hash TEXT NOT NULL, completed_at INTEGER NOT NULL,
                PRIMARY KEY (pot_id, doc, revision, chunk_uri, model_signature)
            )"""
        )
        return connection

    def chunk_done(
        self, *, pot_id: str, doc: str, revision: int, chunk_uri: str,
        model_signature: str, text_hash: str
    ) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT text_hash FROM fact_extraction_chunks WHERE "
                "pot_id=? AND doc=? AND revision=? AND chunk_uri=? AND model_signature=?",
                (pot_id, doc, revision, chunk_uri, model_signature),
            ).fetchone()
        return row is not None and row["text_hash"] == text_hash

    def mark_chunk_done(
        self, *, pot_id: str, doc: str, revision: int, chunk_uri: str,
        model_signature: str, text_hash: str
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO fact_extraction_chunks "
                "(pot_id,doc,revision,chunk_uri,model_signature,text_hash,completed_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT "
                "(pot_id,doc,revision,chunk_uri,model_signature) DO UPDATE SET "
                "text_hash=excluded.text_hash,completed_at=excluded.completed_at",
                (pot_id, doc, revision, chunk_uri, model_signature, text_hash, int(time.time())),
            )

    @staticmethod
    def fingerprint(payload: dict[str, Any]) -> str:
        identity = {
            key: payload.get(key)
            for key in (
                "subject",
                "subject_type",
                "predicate",
                "object",
                "object_type",
                "value",
                "chunk_uri",
                "quote",
            )
        }
        encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def propose(
        self,
        *,
        pot_id: str,
        doc: str,
        revision: int,
        payload: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        fingerprint = self.fingerprint(payload)
        identity = f"{pot_id}\0{doc}\0{revision}\0{fingerprint}"
        item_id = "fact_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        now = int(time.time())
        with self._connect() as db:
            self._expire(db, now)
            cursor = db.execute(
                """INSERT OR IGNORE INTO fact_candidates
                (id, pot_id, doc, revision, fingerprint, status, payload_json,
                 reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
                (
                    item_id,
                    pot_id,
                    doc,
                    revision,
                    fingerprint,
                    json.dumps(payload, sort_keys=True),
                    reason,
                    now,
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM fact_candidates WHERE id=?", (item_id,)
            ).fetchone()
        record = self._record(row)
        record["inserted"] = cursor.rowcount == 1
        return record

    def list(
        self, *, pot_id: str, status: str = "pending", limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._connect() as db:
            self._expire(db, int(time.time()))
            rows = db.execute(
                "SELECT * FROM fact_candidates WHERE pot_id=? AND status=? "
                "ORDER BY created_at DESC, id LIMIT ?",
                (pot_id, status, max(0, min(limit, 1000))),
            ).fetchall()
        return [self._record(row) for row in rows]

    def get(self, *, pot_id: str, item_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM fact_candidates WHERE pot_id=? AND id=?",
                (pot_id, item_id),
            ).fetchone()
        return self._record(row) if row else None

    def transition(
        self, *, pot_id: str, item_id: str, status: str, reason: str | None = None
    ) -> bool:
        if status not in {"verified", "rejected", "closed"}:
            raise ValueError(f"invalid fact Inbox status: {status}")
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE fact_candidates SET status=?, reason=COALESCE(?, reason), "
                "updated_at=? WHERE pot_id=? AND id=? AND status='pending'",
                (status, reason, int(time.time()), pot_id, item_id),
            )
            return cursor.rowcount == 1

    def close_superseded(self, *, pot_id: str, doc: str, revision: int) -> int:
        if not self.path.exists():
            return 0
        with self._connect() as db:
            db.execute(
                "DELETE FROM fact_extraction_chunks WHERE pot_id=? AND doc=? AND revision<?",
                (pot_id, doc, revision),
            )
            cursor = db.execute(
                "UPDATE fact_candidates SET status='closed', reason='superseded by new document revision', "
                "updated_at=? WHERE pot_id=? AND doc=? AND revision<? AND status='pending'",
                (int(time.time()), pot_id, doc, revision),
            )
            return cursor.rowcount

    def clear(self, *, pot_id: str, older_than_days: int) -> int:
        if older_than_days < 0:
            raise ValueError("older-than must be non-negative")
        now = int(time.time())
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE fact_candidates SET status='closed', reason='bulk stale cleanup', "
                "updated_at=? WHERE pot_id=? AND status='pending' AND created_at<=?",
                (now, pot_id, now - older_than_days * 86400),
            )
            return cursor.rowcount

    def prune(self, *, pot_id: str, older_than_days: int = 90) -> int:
        if older_than_days < 0:
            raise ValueError("older-than must be non-negative")
        with self._connect() as db:
            db.execute(
                "DELETE FROM fact_extraction_chunks WHERE pot_id=? AND completed_at<=?",
                (pot_id, int(time.time()) - older_than_days * 86400),
            )
            cursor = db.execute(
                "DELETE FROM fact_candidates WHERE pot_id=? AND status IN ('rejected','closed') "
                "AND updated_at<=?",
                (pot_id, int(time.time()) - older_than_days * 86400),
            )
            return cursor.rowcount

    @staticmethod
    def _expire(db: sqlite3.Connection, now: int) -> None:
        ttl = _ttl_days()
        if ttl:
            db.execute(
                "UPDATE fact_candidates SET status='closed', reason='expired', updated_at=? "
                "WHERE status='pending' AND created_at<?",
                (now, now - ttl * 86400),
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "pot_id": row["pot_id"],
            "doc": row["doc"],
            "revision": row["revision"],
            "status": row["status"],
            "reason": row["reason"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "fact": json.loads(row["payload_json"]),
        }
