"""SQLite registry: dedup tracking + FTS5 chunk index."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_FTS_STOPWORDS = frozenset(
    """a an and are as at be but by did do does for from had has have how in
    into is it its no not of on or than that the their then there these this
    those to was were what when where which who why will with would""".split()
)


def _fts_query_forms(query: str) -> list[str]:
    """FTS5 match expressions to try in order, strictest first.

    The raw query goes first so deliberate FTS5 syntax (phrases, prefix*,
    NEAR) keeps working — a syntax error just skips to the next form. The
    quoted all-terms form keeps exact keyword precision for text FTS5 cannot
    parse; the stopword-filtered any-term form, ranked by bm25, is the last
    resort that lets a natural-language question match a chunk containing
    only its salient words — without stopwords it would match every chunk
    sharing a "the".
    """
    forms: list[str] = []
    raw = query.strip()
    if raw:
        forms.append(raw)
    tokens = _FTS_TOKEN_RE.findall(query)
    quoted = [f'"{t}"' for t in tokens]
    if quoted:
        and_form = " ".join(quoted)
        if and_form != raw:
            forms.append(and_form)
    salient = [f'"{t}"' for t in tokens if t.lower() not in _FTS_STOPWORDS]
    if len(quoted) > 1 and salient:
        or_form = " OR ".join(salient)
        if or_form not in forms:
            forms.append(or_form)
    return forms


@dataclass(slots=True)
class SqliteResourceRegistry:
    db_path: Path

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ingested_files (
                pot_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                imported_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (pot_id, content_hash)
            );

            CREATE TABLE IF NOT EXISTS documents (
                pot_id TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                source_ref TEXT,
                source_kind TEXT,
                revision INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (pot_id, doc_slug)
            );

            CREATE TABLE IF NOT EXISTS sections (
                pot_id TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                content_hash TEXT,
                ordinal INTEGER DEFAULT 0,
                PRIMARY KEY (pot_id, doc_slug, section_slug)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                pot_id TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL,
                seq INTEGER NOT NULL,
                label TEXT,
                chunk_id TEXT,
                content_hash TEXT,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                pot_id UNINDEXED,
                doc_slug UNINDEXED,
                section_slug UNINDEXED,
                seq UNINDEXED,
                content,
                ocr_text
            );

            CREATE TABLE IF NOT EXISTS document_elements (
                pot_id TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                element_id TEXT NOT NULL,
                element_type TEXT,
                text_hash TEXT,
                page_number INTEGER,
                bbox_json TEXT,
                artifact_ref TEXT,
                PRIMARY KEY (pot_id, doc_slug, element_id)
            );

            CREATE TABLE IF NOT EXISTS chunk_provenance (
                pot_id TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL,
                seq INTEGER NOT NULL,
                element_id TEXT NOT NULL,
                page_number INTEGER,
                bbox_json TEXT,
                char_start INTEGER,
                char_end INTEGER,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq, element_id)
            );
            """
        )
        conn.commit()

    def is_file_imported(self, pot_id: str, content_hash: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ingested_files WHERE pot_id = ? AND content_hash = ?",
                (pot_id, content_hash),
            ).fetchone()
            return row is not None

    def record_file_hash(self, pot_id: str, content_hash: str, doc_slug: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ingested_files (pot_id, content_hash, doc_slug)
                VALUES (?, ?, ?)
                ON CONFLICT(pot_id, content_hash) DO UPDATE SET
                    doc_slug = excluded.doc_slug,
                    imported_at = datetime('now')
                """,
                (pot_id, content_hash, doc_slug),
            )
            conn.commit()

    def upsert_document(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        source_ref: str,
        source_kind: str,
    ) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT revision FROM documents WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            ).fetchone()
            revision = (int(row["revision"]) + 1) if row else 1
            conn.execute(
                """
                INSERT INTO documents (pot_id, doc_slug, source_ref, source_kind, revision, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(pot_id, doc_slug) DO UPDATE SET
                    source_ref = excluded.source_ref,
                    source_kind = excluded.source_kind,
                    revision = excluded.revision,
                    updated_at = datetime('now')
                """,
                (pot_id, doc_slug, source_ref, source_kind, revision),
            )
            conn.commit()
            return revision

    def replace_sections(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        sections: list[dict[str, Any]],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return (added, kept, changed, removed) section slugs."""
        with self.connect() as conn:
            existing = {
                row["section_slug"]: dict(row)
                for row in conn.execute(
                    "SELECT * FROM sections WHERE pot_id = ? AND doc_slug = ?",
                    (pot_id, doc_slug),
                )
            }
            new_slugs = {s["slug"] for s in sections}
            removed = sorted(set(existing) - new_slugs)
            added: list[str] = []
            kept: list[str] = []
            changed: list[str] = []

            for slug in removed:
                conn.execute(
                    "DELETE FROM chunk_provenance WHERE pot_id = ? AND doc_slug = ? AND section_slug = ?",
                    (pot_id, doc_slug, slug),
                )
                conn.execute(
                    "DELETE FROM chunks WHERE pot_id = ? AND doc_slug = ? AND section_slug = ?",
                    (pot_id, doc_slug, slug),
                )
                conn.execute(
                    "DELETE FROM sections WHERE pot_id = ? AND doc_slug = ? AND section_slug = ?",
                    (pot_id, doc_slug, slug),
                )

            for section in sections:
                slug = section["slug"]
                prev = existing.get(slug)
                if prev is None:
                    added.append(slug)
                elif prev.get("content_hash") == section.get("content_hash"):
                    kept.append(slug)
                else:
                    changed.append(slug)

                conn.execute(
                    """
                    INSERT INTO sections (
                        pot_id, doc_slug, section_slug, title, summary, content_hash, ordinal
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pot_id, doc_slug, section_slug) DO UPDATE SET
                        title = excluded.title,
                        summary = excluded.summary,
                        content_hash = excluded.content_hash,
                        ordinal = excluded.ordinal
                    """,
                    (
                        pot_id,
                        doc_slug,
                        slug,
                        section.get("title"),
                        section.get("summary"),
                        section.get("content_hash"),
                        section.get("ordinal", 0),
                    ),
                )

            conn.commit()
            return added, kept, changed, removed

    def list_element_ids(self, pot_id: str, doc_slug: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT element_id FROM document_elements
                WHERE pot_id = ? AND doc_slug = ?
                ORDER BY element_id
                """,
                (pot_id, doc_slug),
            ).fetchall()
            return [str(row["element_id"]) for row in rows]

    def replace_elements(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        elements: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        """Return (added, removed) element ids."""
        import json

        with self.connect() as conn:
            existing = {
                row["element_id"]
                for row in conn.execute(
                    "SELECT element_id FROM document_elements WHERE pot_id = ? AND doc_slug = ?",
                    (pot_id, doc_slug),
                )
            }
            new_ids = {e["element_id"] for e in elements}
            removed = sorted(existing - new_ids)
            added = sorted(new_ids - existing)

            for element_id in removed:
                conn.execute(
                    "DELETE FROM chunk_provenance WHERE pot_id = ? AND doc_slug = ? AND element_id = ?",
                    (pot_id, doc_slug, element_id),
                )
                conn.execute(
                    "DELETE FROM document_elements WHERE pot_id = ? AND doc_slug = ? AND element_id = ?",
                    (pot_id, doc_slug, element_id),
                )

            for element in elements:
                bbox_json = json.dumps(element.get("bbox")) if element.get("bbox") else None
                conn.execute(
                    """
                    INSERT INTO document_elements (
                        pot_id, doc_slug, element_id, element_type, text_hash,
                        page_number, bbox_json, artifact_ref
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pot_id, doc_slug, element_id) DO UPDATE SET
                        element_type = excluded.element_type,
                        text_hash = excluded.text_hash,
                        page_number = excluded.page_number,
                        bbox_json = excluded.bbox_json,
                        artifact_ref = excluded.artifact_ref
                    """,
                    (
                        pot_id,
                        doc_slug,
                        element["element_id"],
                        element.get("element_type"),
                        element.get("text_hash"),
                        element.get("page_number"),
                        bbox_json,
                        element.get("artifact_ref"),
                    ),
                )
            conn.commit()
            return added, removed

    def replace_chunk_provenance(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
        rows: list[dict[str, Any]],
    ) -> None:
        import json

        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM chunk_provenance
                WHERE pot_id = ? AND doc_slug = ? AND section_slug = ? AND seq = ?
                """,
                (pot_id, doc_slug, section_slug, seq),
            )
            for row in rows:
                bbox_json = json.dumps(row.get("bbox")) if row.get("bbox") else None
                conn.execute(
                    """
                    INSERT INTO chunk_provenance (
                        pot_id, doc_slug, section_slug, seq, element_id,
                        page_number, bbox_json, char_start, char_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pot_id,
                        doc_slug,
                        section_slug,
                        seq,
                        row["element_id"],
                        row.get("page_number"),
                        bbox_json,
                        row.get("char_start"),
                        row.get("char_end"),
                    ),
                )
            conn.commit()

    def get_chunk_provenance(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
    ) -> list[dict[str, Any]]:
        import json

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT element_id, page_number, bbox_json, char_start, char_end
                FROM chunk_provenance
                WHERE pot_id = ? AND doc_slug = ? AND section_slug = ? AND seq = ?
                ORDER BY element_id
                """,
                (pot_id, doc_slug, section_slug, seq),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
                result.append(
                    {
                        "element_id": row["element_id"],
                        "page_number": row["page_number"],
                        "bbox": bbox,
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                    }
                )
            return result

    def replace_section_chunks(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunk_provenance WHERE pot_id = ? AND doc_slug = ? AND section_slug = ?",
                (pot_id, doc_slug, section_slug),
            )
            conn.execute(
                "DELETE FROM chunks WHERE pot_id = ? AND doc_slug = ? AND section_slug = ?",
                (pot_id, doc_slug, section_slug),
            )
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (
                        pot_id, doc_slug, section_slug, seq, label, chunk_id, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pot_id,
                        doc_slug,
                        section_slug,
                        chunk["seq"],
                        chunk.get("label"),
                        chunk.get("chunk_id"),
                        chunk.get("content_hash"),
                    ),
                )
            conn.commit()

    def rebuild_fts_for_document(self, pot_id: str, doc_slug: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunks_fts WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            rows = conn.execute(
                """
                SELECT c.pot_id, c.doc_slug, c.section_slug, c.seq,
                       ch.content, ch.ocr_text
                FROM chunks c
                INNER JOIN chunk_text ch ON
                    ch.pot_id = c.pot_id AND ch.doc_slug = c.doc_slug
                    AND ch.section_slug = c.section_slug AND ch.seq = c.seq
                WHERE c.pot_id = ? AND c.doc_slug = ?
                """,
                (pot_id, doc_slug),
            ).fetchall()
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO chunks_fts (
                        pot_id, doc_slug, section_slug, seq, content, ocr_text
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["pot_id"],
                        row["doc_slug"],
                        row["section_slug"],
                        row["seq"],
                        row["content"] or "",
                        row["ocr_text"] or "",
                    ),
                )
            conn.commit()

    def upsert_chunk_text(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
        content: str,
        ocr_text: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_text (
                    pot_id TEXT NOT NULL,
                    doc_slug TEXT NOT NULL,
                    section_slug TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    ocr_text TEXT,
                    PRIMARY KEY (pot_id, doc_slug, section_slug, seq)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO chunk_text (pot_id, doc_slug, section_slug, seq, content, ocr_text)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pot_id, doc_slug, section_slug, seq) DO UPDATE SET
                    content = excluded.content,
                    ocr_text = excluded.ocr_text
                """,
                (pot_id, doc_slug, section_slug, seq, content, ocr_text),
            )
            conn.commit()

    def search_chunks(
        self, pot_id: str, query: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        forms = _fts_query_forms(query)
        if not forms:
            return []
        with self.connect() as conn:
            for fts_query in forms:
                try:
                    rows = conn.execute(
                        """
                        SELECT pot_id, doc_slug, section_slug, seq, bm25(chunks_fts) AS score
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ? AND pot_id = ?
                        ORDER BY score
                        LIMIT ?
                        """,
                        (fts_query, pot_id, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                if rows:
                    return [dict(row) for row in rows]
        return []

    def list_documents(self, pot_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT d.doc_slug, d.source_ref, d.source_kind, d.revision, d.updated_at,
                       COUNT(s.section_slug) AS section_count
                FROM documents d
                LEFT JOIN sections s ON
                    s.pot_id = d.pot_id AND s.doc_slug = d.doc_slug
                WHERE d.pot_id = ?
                GROUP BY d.pot_id, d.doc_slug
                ORDER BY d.updated_at DESC
                """,
                (pot_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def remove_document(self, pot_id: str, doc_slug: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunks_fts WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM chunk_provenance WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM document_elements WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM chunk_text WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM chunks WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM sections WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM documents WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM ingested_files WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.commit()

    def get_section_summaries(self, pot_id: str, doc_slug: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT section_slug, title, summary, content_hash, ordinal
                FROM sections
                WHERE pot_id = ? AND doc_slug = ?
                ORDER BY ordinal, section_slug
                """,
                (pot_id, doc_slug),
            ).fetchall()
            return [dict(row) for row in rows]


__all__ = ["SqliteResourceRegistry"]
