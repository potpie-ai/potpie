"""Durable immutable preview metadata, separate from the rebuildable index."""

import asyncio
import json
import sqlite3
from pathlib import Path

from potpie_context_core.graph_journal import (
    JournalError,
    decode_journal,
    journal_hash,
    journal_json,
)


class LocalRollbackPreviews:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rollback_previews (id TEXT PRIMARY KEY, pot TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL)"
            )

    async def put_async(self, preview, request):
        raw, digest = journal_json((preview, request)), journal_hash((preview, request))

        def write():
            with sqlite3.connect(self.path, timeout=30) as conn:
                conn.execute(
                    "INSERT INTO rollback_previews VALUES (?,?,?,?)",
                    (preview.preview_id, preview.pot_id, raw, digest),
                )

        await asyncio.to_thread(write)

    async def get_async(self, *, preview_id, pot_id):
        def read():
            with sqlite3.connect(self.path, timeout=30) as conn:
                row = conn.execute(
                    "SELECT payload,hash FROM rollback_previews WHERE id=? AND pot=?",
                    (preview_id, pot_id),
                ).fetchone()
            if row is None:
                return None
            result = decode_journal(json.loads(row[0]))
            if journal_hash(result) != row[1]:
                raise JournalError("persisted rollback preview hash mismatch")
            return result

        return await asyncio.to_thread(read)
