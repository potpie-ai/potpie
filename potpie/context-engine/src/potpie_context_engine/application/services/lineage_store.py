"""SQLite payload + span index for generation lineage.

Trustgraph-style: IDs and edges live on the graph; full prompt/spec text and
file+line lookups live here. Default location:

    ~/.potpie/lineage/<pot>/registry.db
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS payloads (
    hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (hash, kind)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_key TEXT PRIMARY KEY,
    harness TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS latest_prompt (
    session_key TEXT PRIMARY KEY,
    prompt_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    code_asset_key TEXT NOT NULL,
    prompt_hash TEXT,
    spec_hash TEXT,
    session_key TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_spans_path ON spans(path, line_start, line_end);
"""

_SQLITE_BUSY_TIMEOUT_SECONDS = 10.0
_SQLITE_BUSY_TIMEOUT_MS = int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_lineage_db_path(pot_id: str, *, home: Path | None = None) -> Path:
    from potpie_context_engine.adapters.outbound.local_paths import default_home

    root = home if home is not None else default_home()
    safe = (pot_id or "default").replace("/", "__").replace(":", "_")
    return root / "lineage" / safe / "registry.db"


@dataclass(frozen=True, slots=True)
class SpanHit:
    path: str
    line_start: int
    line_end: int
    code_asset_key: str
    prompt_hash: str | None
    spec_hash: str | None
    session_key: str | None
    created_at: str
    prompt_text: str | None = None
    spec_text: str | None = None


class LineageStore:
    """File+line → CodeAsset key and prompt/spec payload lookup."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def put_payload(self, *, hash_: str, kind: str, text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO payloads(hash, kind, text, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(hash, kind) DO NOTHING
                """,
                (hash_, kind, text, _utcnow()),
            )

    def get_payload(self, hash_: str, kind: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text FROM payloads WHERE hash = ? AND kind = ?",
                (hash_, kind),
            ).fetchone()
        return None if row is None else str(row["text"])

    def put_session(self, *, session_key: str, harness: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_key, harness, session_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_key) DO NOTHING
                """,
                (session_key, harness, session_id, _utcnow()),
            )

    def remember_prompt(self, *, session_key: str, prompt_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO latest_prompt(session_key, prompt_hash, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    prompt_hash = excluded.prompt_hash,
                    updated_at = excluded.updated_at
                """,
                (session_key, prompt_hash, _utcnow()),
            )

    def latest_prompt_hash(self, session_key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT prompt_hash FROM latest_prompt WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return None if row is None else str(row["prompt_hash"])

    def put_span(
        self,
        *,
        path: str,
        line_start: int,
        line_end: int,
        code_asset_key: str,
        prompt_hash: str | None,
        spec_hash: str | None,
        session_key: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO spans(
                    path, line_start, line_end, code_asset_key,
                    prompt_hash, spec_hash, session_key, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    int(line_start),
                    int(line_end),
                    code_asset_key,
                    prompt_hash,
                    spec_hash,
                    session_key,
                    _utcnow(),
                ),
            )

    def overlapping_spans(
        self, *, path: str, line_start: int, line_end: int
    ) -> list[SpanHit]:
        normalized = _normalize_path(path)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT path, line_start, line_end, code_asset_key,
                       prompt_hash, spec_hash, session_key, created_at
                FROM spans
                WHERE (path = ? OR path LIKE ?)
                  AND line_start <= ?
                  AND line_end >= ?
                ORDER BY created_at DESC
                """,
                (normalized, f"%/{normalized}", int(line_end), int(line_start)),
            ).fetchall()
        hits: list[SpanHit] = []
        for row in rows:
            prompt_hash = row["prompt_hash"]
            spec_hash = row["spec_hash"]
            hits.append(
                SpanHit(
                    path=str(row["path"]),
                    line_start=int(row["line_start"]),
                    line_end=int(row["line_end"]),
                    code_asset_key=str(row["code_asset_key"]),
                    prompt_hash=str(prompt_hash) if prompt_hash else None,
                    spec_hash=str(spec_hash) if spec_hash else None,
                    session_key=str(row["session_key"]) if row["session_key"] else None,
                    created_at=str(row["created_at"]),
                    prompt_text=(
                        self.get_payload(str(prompt_hash), "prompt")
                        if prompt_hash
                        else None
                    ),
                    spec_text=(
                        self.get_payload(str(spec_hash), "spec") if spec_hash else None
                    ),
                )
            )
        return hits


def _normalize_path(path: str) -> str:
    return normalize_lineage_path(path)


def normalize_lineage_path(path: str) -> str:
    """Canonicalize separators and remove only an exact leading ``./``."""
    return str(path).replace("\\", "/").strip().removeprefix("./")


def payload_hash(text: str) -> str:
    """Match PromptTurn/SpecRequirement CONTENT_HASH body (12 hex chars)."""
    from potpie_context_engine.core.identity import mint_entity_key, get_identity

    key = mint_entity_key(get_identity("PromptTurn"), content=text)
    return key.rsplit(":", 1)[-1]


__all__ = [
    "LineageStore",
    "SpanHit",
    "default_lineage_db_path",
    "normalize_lineage_path",
    "payload_hash",
]
