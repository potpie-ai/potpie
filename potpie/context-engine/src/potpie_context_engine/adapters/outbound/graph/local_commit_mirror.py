"""Rebuildable SQLite commit listing index; never stores graph snapshots."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from potpie_context_core.graph_journal import JournalError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commits (
    pot TEXT NOT NULL, generation TEXT NOT NULL, sequence INTEGER NOT NULL,
    commit_id TEXT NOT NULL, actor TEXT NOT NULL, origin TEXT NOT NULL,
    header TEXT NOT NULL, PRIMARY KEY(pot,generation,sequence), UNIQUE(pot,commit_id)
);
CREATE INDEX IF NOT EXISTS commits_actor ON commits(pot,generation,actor,sequence DESC);
CREATE INDEX IF NOT EXISTS commits_origin ON commits(pot,generation,origin,sequence DESC);
CREATE TABLE IF NOT EXISTS progress (
    pot TEXT NOT NULL, generation TEXT NOT NULL, sequence INTEGER NOT NULL,
    PRIMARY KEY(pot,generation)
);
"""


class LocalCommitMirror:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    async def reset_for_rebuild_async(self, *, pot_id, generation):
        def reset():
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "DELETE FROM commits WHERE pot=? AND generation=?",
                    (pot_id, generation),
                )
                conn.execute(
                    "DELETE FROM progress WHERE pot=? AND generation=?",
                    (pot_id, generation),
                )

        await asyncio.to_thread(reset)

    async def progress_async(self, *, pot_id, generation):
        def read():
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT sequence FROM progress WHERE pot=? AND generation=?",
                    (pot_id, generation),
                ).fetchone()
                return row[0] if row else 0

        return await asyncio.to_thread(read)

    async def header_at_async(self, *, pot_id, generation, sequence):
        def read():
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT header FROM commits WHERE pot=? AND generation=? AND sequence=?",
                    (pot_id, generation, sequence),
                ).fetchone()
                return json.loads(row[0]) if row else None

        return await asyncio.to_thread(read)

    async def append_batch_async(self, *, headers, expected_progress):
        if not headers or not 1 <= len(headers) <= 200 or expected_progress < 0:
            raise JournalError("invalid mirror batch")
        pot, generation = headers[0]["pot_id"], headers[0]["journal_generation"]
        for sequence, header in enumerate(headers, expected_progress + 1):
            if (header["pot_id"], header["journal_generation"], header["sequence"]) != (
                pot,
                generation,
                sequence,
            ):
                raise JournalError("mirror batch must be contiguous and pot scoped")

        def write():
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT sequence FROM progress WHERE pot=? AND generation=?",
                    (pot, generation),
                ).fetchone()
                progress = row[0] if row else 0
                if progress < expected_progress:
                    raise JournalError("mirror checkpoint would skip a sequence")
                for header in headers:
                    raw = json.dumps(
                        header, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    if len(raw.encode()) > 8_000_000:
                        raise JournalError("commit header exceeds mirror byte limit")
                    conn.execute(
                        "INSERT OR IGNORE INTO commits VALUES (?,?,?,?,?,?,?)",
                        (
                            pot,
                            generation,
                            header["sequence"],
                            header["commit_id"],
                            header["actor"],
                            header["origin"],
                            raw,
                        ),
                    )
                    stored = conn.execute(
                        "SELECT header FROM commits WHERE pot=? AND generation=? AND sequence=?",
                        (pot, generation, header["sequence"]),
                    ).fetchone()
                    if not stored or json.loads(stored[0]) != header:
                        raise JournalError(
                            "immutable duplicate receipt/header mismatch"
                        )
                conn.execute(
                    "INSERT INTO progress VALUES (?,?,?) ON CONFLICT(pot,generation) "
                    "DO UPDATE SET sequence=excluded.sequence",
                    (pot, generation, max(progress, headers[-1]["sequence"])),
                )

        await asyncio.to_thread(write)

    async def read_page_async(
        self,
        *,
        pot_id,
        generation,
        before_sequence,
        ceiling,
        actor=None,
        origin=None,
        logical_key=None,
        limit=51,
    ):
        if not 1 <= limit <= 201:
            raise JournalError("invalid header page size")

        def read():
            query = "SELECT header FROM commits WHERE pot=? AND generation=? AND sequence<? AND sequence<=?"
            params = [pot_id, generation, before_sequence, ceiling]
            for name, value in (("actor", actor), ("origin", origin)):
                if value is not None:
                    query += f" AND {name}=?"
                    params.append(value)
            if logical_key is not None:
                query += " AND EXISTS (SELECT 1 FROM json_each(header,'$.touched_keys') WHERE value=?)"
                params.append(logical_key)
            query += " ORDER BY sequence DESC LIMIT ?"
            params.append(limit)
            with self._connect() as conn:
                conn.execute("BEGIN")
                checkpoint = conn.execute(
                    "SELECT sequence FROM progress WHERE pot=? AND generation=?",
                    (pot_id, generation),
                ).fetchone()
                rows = tuple(json.loads(row[0]) for row in conn.execute(query, params))
                return (checkpoint[0] if checkpoint else 0), rows

        return await asyncio.to_thread(read)

    async def list_headers_async(self, **kwargs):
        _, rows = await self.read_page_async(**kwargs)
        return rows
