"""SQLite registry: dedup tracking + FTS5 chunk index."""

from __future__ import annotations

import re
import sqlite3
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from potpie_context_engine.core.ports.resource_index import ResourceSearchFilters

_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_FTS_STOPWORDS = frozenset(
    """a an and are as at be but by did do does for from had has have how in
    into is it its no not of on or than that the their then there these this
    those to was were what when where which who why will with would""".split()
)
_QUERY_SYNONYMS = {
    "excluded": ("removed", "omitted"),
    "omitted": ("removed", "excluded"),
    "prevent": ("avoid", "block"),
    "approximately": ("about", "roughly"),
    "performance": ("accuracy", "score"),
    "underfits": ("underfitting", "underfit"),
    "underfitting": ("underfits", "underfit"),
    "encyclopedia": ("wikipedia",),
    "wikipedia": ("encyclopedia",),
}
_FACET_NOISE = frozenset(
    "combine statement headline evaluation result how much used many reached "
    "question paper still pieces evidence support"
    .split()
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
        # Preserve deliberate FTS5 syntax, but do not treat a long natural
        # language question as an all-token FTS expression. The latter lets a
        # generic passage matching many question words outrank a focused
        # numeric/table facet.
        if len(_FTS_TOKEN_RE.findall(raw)) <= 2 or re.search(
            r'["*]|\b(?:NEAR|OR|AND|NOT)\b', raw
        ):
            forms.append(raw)
    tokens = _FTS_TOKEN_RE.findall(query)
    quoted = [f'"{t}"' for t in tokens]
    if quoted:
        and_form = " ".join(quoted)
        if and_form != raw:
            forms.append(and_form)
    salient = [f'"{t}"' for t in tokens if t.lower() not in _FTS_STOPWORDS]
    multi_faceted = len(_query_facets(query)) > 1
    if len(quoted) > 1 and salient and not multi_faceted:
        or_form = " OR ".join(salient)
        if or_form not in forms:
            forms.append(or_form)
        expanded: list[str] = []
        for token in tokens:
            if token.lower() in _FTS_STOPWORDS:
                continue
            expanded.append(f'"{token}"')
            expanded.extend(
                f'"{variant}"'
                for variant in _QUERY_SYNONYMS.get(token.lower(), ())
            )
        expanded_form = " OR ".join(dict.fromkeys(expanded))
        if expanded_form and expanded_form not in forms:
            forms.append(expanded_form)
    for facet in _query_facets(query):
        facet_tokens = _FTS_TOKEN_RE.findall(facet)
        facet_salient = [
            token
            for token in facet_tokens
            if token.lower() not in _FTS_STOPWORDS
            and token.lower() not in _FACET_NOISE
        ]
        if len(facet_salient) < 2:
            continue
        facet_form = " ".join(f'"{token}"' for token in facet_salient)
        if facet_form not in forms:
            forms.append(facet_form)
        expanded_facet_tokens: list[str] = []
        for token in facet_salient:
            expanded_facet_tokens.append(token)
            expanded_facet_tokens.extend(_QUERY_SYNONYMS.get(token.lower(), ()))
        facet_or_form = " OR ".join(
            f'"{token}"' for token in dict.fromkeys(expanded_facet_tokens)
        )
        if facet_or_form not in forms:
            forms.append(facet_or_form)
    joined_units = re.sub(
        r"\b(\d+(?:\.\d+)?)\s+([kmgt](?:b)?|percent|%)\b",
        lambda match: f"{match.group(1)}{match.group(2)}",
        query,
        flags=re.IGNORECASE,
    )
    if joined_units != query:
        forms.append(f'"{joined_units.strip()}"')
    if re.search(r"\bhow\s+(much|large)\b", query, flags=re.IGNORECASE) and re.search(
        r"\b(data|dataset|size)\b", query, flags=re.IGNORECASE
    ):
        forms.append('"million" OR "GB" OR "documents"')
    return forms


def _query_facets(query: str) -> list[str]:
    """Split natural-language questions into independently searchable facts."""
    return [
        facet.strip()
        for facet in re.split(r"\band\b|[;|]", query, flags=re.IGNORECASE)
        if facet.strip()
    ]


def _is_title_query(query: str) -> bool:
    return bool(re.search(r"\b(title|heading|name)\b", query, flags=re.IGNORECASE))


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

            CREATE TABLE IF NOT EXISTS chunk_text (
                pot_id TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL,
                seq INTEGER NOT NULL,
                section_title TEXT NOT NULL DEFAULT '',
                source_ref TEXT NOT NULL DEFAULT '',
                source_kind TEXT NOT NULL DEFAULT '',
                parser_tier TEXT NOT NULL DEFAULT '',
                page_numbers TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                ocr_text TEXT,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq)
            );

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                pot_id TEXT NOT NULL,
                doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL,
                seq INTEGER NOT NULL,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                pot_id UNINDEXED,
                doc_slug UNINDEXED,
                section_slug UNINDEXED,
                seq UNINDEXED,
                source_ref UNINDEXED,
                source_kind UNINDEXED,
                parser_tier UNINDEXED,
                page_numbers UNINDEXED,
                section_title,
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
                metadata_json TEXT,
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
        conn.execute("BEGIN IMMEDIATE")
        migrated = False
        chunk_text_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(chunk_text)")
        }
        for column in (
            "section_title",
            "source_ref",
            "source_kind",
            "parser_tier",
            "page_numbers",
        ):
            if column not in chunk_text_columns:
                conn.execute(
                    f"ALTER TABLE chunk_text ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                )
                migrated = True

        document_element_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(document_elements)")
        }
        if "metadata_json" not in document_element_columns:
            conn.execute(
                "ALTER TABLE document_elements ADD COLUMN metadata_json TEXT"
            )
            migrated = True

        expected_fts_columns = [
            "pot_id",
            "doc_slug",
            "section_slug",
            "seq",
            "source_ref",
            "source_kind",
            "parser_tier",
            "page_numbers",
            "section_title",
            "content",
            "ocr_text",
        ]
        actual_fts_columns = [
            str(row[1]) for row in conn.execute("PRAGMA table_info(chunks_fts)")
        ]
        if actual_fts_columns != expected_fts_columns:
            conn.execute("DROP TABLE chunks_fts")
            conn.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    pot_id UNINDEXED,
                    doc_slug UNINDEXED,
                    section_slug UNINDEXED,
                    seq UNINDEXED,
                    source_ref UNINDEXED,
                    source_kind UNINDEXED,
                    parser_tier UNINDEXED,
                    page_numbers UNINDEXED,
                    section_title,
                    content,
                    ocr_text
                )
                """
            )
            migrated = True

        if migrated:
            conn.execute(
                """
                UPDATE chunk_text
                SET section_title = COALESCE((
                        SELECT s.title FROM sections s
                        WHERE s.pot_id = chunk_text.pot_id
                          AND s.doc_slug = chunk_text.doc_slug
                          AND s.section_slug = chunk_text.section_slug
                    ), ''),
                    source_ref = COALESCE((
                        SELECT d.source_ref FROM documents d
                        WHERE d.pot_id = chunk_text.pot_id
                          AND d.doc_slug = chunk_text.doc_slug
                    ), ''),
                    source_kind = COALESCE((
                        SELECT d.source_kind FROM documents d
                        WHERE d.pot_id = chunk_text.pot_id
                          AND d.doc_slug = chunk_text.doc_slug
                    ), '')
                """
            )
            for row in conn.execute(
                """
                SELECT pot_id, doc_slug, section_slug, seq,
                       group_concat(DISTINCT page_number) AS pages
                FROM chunk_provenance
                WHERE page_number IS NOT NULL
                GROUP BY pot_id, doc_slug, section_slug, seq
                """
            ).fetchall():
                conn.execute(
                    """
                    UPDATE chunk_text SET page_numbers = ?
                    WHERE pot_id = ? AND doc_slug = ? AND section_slug = ? AND seq = ?
                    """,
                    (
                        f",{row['pages']},",
                        row["pot_id"],
                        row["doc_slug"],
                        row["section_slug"],
                        row["seq"],
                    ),
                )
            try:
                self._rebuild_fts_rows(conn)
            except Exception:
                conn.rollback()
                raise
        conn.commit()

    @staticmethod
    def _rebuild_fts_rows(
        conn: sqlite3.Connection,
        *,
        pot_id: str | None = None,
        doc_slug: str | None = None,
    ) -> None:
        if pot_id is None:
            conn.execute("DELETE FROM chunks_fts")
            where = ""
            params: tuple[Any, ...] = ()
        else:
            conn.execute(
                "DELETE FROM chunks_fts WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            where = "WHERE pot_id = ? AND doc_slug = ?"
            params = (pot_id, doc_slug)
        rows = conn.execute(
            f"""
            SELECT pot_id, doc_slug, section_slug, seq, source_ref, source_kind,
                   parser_tier, page_numbers, section_title, content, ocr_text
            FROM chunk_text
            {where}
            """,
            params,
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                INSERT INTO chunks_fts (
                    pot_id, doc_slug, section_slug, seq, source_ref, source_kind,
                    parser_tier, page_numbers, section_title, content, ocr_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row),
            )

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
                bbox_json = (
                    json.dumps(element.get("bbox")) if element.get("bbox") else None
                )
                metadata = {
                    key: element.get(key)
                    for key in (
                        "caption",
                        "ocr_text",
                        "visual_caption",
                        "figure_number",
                        "table_number",
                    )
                    if element.get(key) not in (None, "")
                }
                conn.execute(
                    """
                    INSERT INTO document_elements (
                        pot_id, doc_slug, element_id, element_type, text_hash,
                        page_number, bbox_json, artifact_ref, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pot_id, doc_slug, element_id) DO UPDATE SET
                        element_type = excluded.element_type,
                        text_hash = excluded.text_hash,
                        page_number = excluded.page_number,
                        bbox_json = excluded.bbox_json,
                        artifact_ref = excluded.artifact_ref,
                        metadata_json = excluded.metadata_json
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
                        json.dumps(metadata, separators=(",", ":")),
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
                SELECT cp.element_id, cp.page_number, cp.bbox_json,
                       cp.char_start, cp.char_end,
                       de.element_type, de.artifact_ref, de.metadata_json
                FROM chunk_provenance cp
                LEFT JOIN document_elements de
                  ON de.pot_id = cp.pot_id
                 AND de.doc_slug = cp.doc_slug
                 AND de.element_id = CASE
                       WHEN instr(cp.element_id, '#') > 0
                       THEN substr(cp.element_id, 1, instr(cp.element_id, '#') - 1)
                       ELSE cp.element_id
                     END
                WHERE cp.pot_id = ? AND cp.doc_slug = ?
                  AND cp.section_slug = ? AND cp.seq = ?
                ORDER BY cp.element_id
                """,
                (pot_id, doc_slug, section_slug, seq),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
                metadata = (
                    json.loads(row["metadata_json"])
                    if row["metadata_json"]
                    else {}
                )
                result.append(
                    {
                        "element_id": row["element_id"],
                        "page_number": row["page_number"],
                        "bbox": bbox,
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                        "element_type": row["element_type"],
                        "artifact_ref": row["artifact_ref"],
                        **metadata,
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
            self._rebuild_fts_rows(conn, pot_id=pot_id, doc_slug=doc_slug)
            conn.commit()

    def replace_chunk_text_document(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        rows: list[dict[str, Any]],
        embeddings: dict[tuple[str, int], tuple[str, list[float]]] | None = None,
    ) -> int:
        """Replace one document's index plane in a single SQLite transaction."""
        with self.connect() as conn:
            try:
                conn.execute(
                    "DELETE FROM chunks_fts WHERE pot_id = ? AND doc_slug = ?",
                    (pot_id, doc_slug),
                )
                conn.execute(
                    "DELETE FROM chunk_text WHERE pot_id = ? AND doc_slug = ?",
                    (pot_id, doc_slug),
                )
                conn.execute(
                    "DELETE FROM chunk_embeddings WHERE pot_id = ? AND doc_slug = ?",
                    (pot_id, doc_slug),
                )
                for row in rows:
                    conn.execute(
                        """
                        INSERT INTO chunk_text (
                            pot_id, doc_slug, section_slug, seq, section_title,
                            source_ref, source_kind, parser_tier, page_numbers,
                            content, ocr_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pot_id,
                            doc_slug,
                            row["section_slug"],
                            row["seq"],
                            row.get("section_title", ""),
                            row.get("source_ref", ""),
                            row.get("source_kind", ""),
                            row.get("parser_tier", ""),
                            (
                                ""
                                if not row.get("page_numbers")
                                else f",{','.join(map(str, row['page_numbers']))},"
                            ),
                            row.get("content", ""),
                            row.get("ocr_text", ""),
                        ),
                    )
                for (section_slug, seq), (model, vector) in (embeddings or {}).items():
                    conn.execute(
                        """
                        INSERT INTO chunk_embeddings (
                            pot_id, doc_slug, section_slug, seq, model, vector_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pot_id,
                            doc_slug,
                            section_slug,
                            seq,
                            model,
                            json.dumps(vector, separators=(",", ":")),
                        ),
                    )
                self._rebuild_fts_rows(conn, pot_id=pot_id, doc_slug=doc_slug)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return len(rows)

    def search_semantic_chunks(
        self,
        pot_id: str,
        query_vector: tuple[float, ...],
        *,
        limit: int = 20,
        filters: ResourceSearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        """Search persisted chunk vectors with deterministic cosine similarity."""
        if not query_vector:
            return []
        active = filters or ResourceSearchFilters()
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm <= 0:
            return []
        page_pattern = (
            f"%,{active.page_number},%" if active.page_number is not None else None
        )
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT pot_id, doc_slug, section_slug, seq, model, vector_json
                FROM chunk_embeddings
                WHERE pot_id = ?
                  AND (? IS NULL OR doc_slug = ?)
                  AND (? IS NULL OR EXISTS (
                      SELECT 1 FROM chunk_text AS ct
                      WHERE ct.pot_id = chunk_embeddings.pot_id
                        AND ct.doc_slug = chunk_embeddings.doc_slug
                        AND ct.section_slug = chunk_embeddings.section_slug
                        AND ct.seq = chunk_embeddings.seq
                        AND ct.source_kind = ?
                  ))
                  AND (? IS NULL OR EXISTS (
                      SELECT 1 FROM chunk_text AS ct
                      WHERE ct.pot_id = chunk_embeddings.pot_id
                        AND ct.doc_slug = chunk_embeddings.doc_slug
                        AND ct.section_slug = chunk_embeddings.section_slug
                        AND ct.seq = chunk_embeddings.seq
                        AND ct.page_numbers LIKE ?
                  ))
                """,
                (
                    pot_id,
                    active.doc_slug,
                    active.doc_slug,
                    active.source_kind,
                    active.source_kind,
                    page_pattern,
                    page_pattern,
                ),
            ).fetchall()

        scored: list[dict[str, Any]] = []
        for row in rows:
            try:
                vector = [float(value) for value in json.loads(row["vector_json"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if len(vector) != len(query_vector):
                continue
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if vector_norm <= 0:
                continue
            similarity = sum(
                left * right for left, right in zip(query_vector, vector, strict=True)
            ) / (query_norm * vector_norm)
            scored.append(
                {
                    "pot_id": row["pot_id"],
                    "doc_slug": row["doc_slug"],
                    "section_slug": row["section_slug"],
                    "seq": row["seq"],
                    "semantic_score": similarity,
                    "score": similarity,
                }
            )
        scored.sort(
            key=lambda row: (
                -float(row["semantic_score"]),
                row["doc_slug"],
                row["section_slug"],
                row["seq"],
            )
        )
        return scored[:limit]

    def upsert_chunk_text(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
        content: str,
        ocr_text: str = "",
        section_title: str = "",
        source_ref: str = "",
        source_kind: str = "",
        parser_tier: str = "",
        page_numbers: list[int] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chunk_text (
                    pot_id, doc_slug, section_slug, seq, section_title,
                    source_ref, source_kind, parser_tier, page_numbers,
                    content, ocr_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pot_id, doc_slug, section_slug, seq) DO UPDATE SET
                    section_title = excluded.section_title,
                    source_ref = excluded.source_ref,
                    source_kind = excluded.source_kind,
                    parser_tier = excluded.parser_tier,
                    page_numbers = excluded.page_numbers,
                    content = excluded.content,
                    ocr_text = excluded.ocr_text
                """,
                (
                    pot_id,
                    doc_slug,
                    section_slug,
                    seq,
                    section_title,
                    source_ref,
                    source_kind,
                    parser_tier,
                    "" if not page_numbers else f",{','.join(map(str, page_numbers))},",
                    content,
                    ocr_text,
                ),
            )
            conn.commit()

    def search_chunks(
        self,
        pot_id: str,
        query: str,
        limit: int = 20,
        filters: ResourceSearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        forms = _fts_query_forms(query)
        if not forms:
            return []
        active = filters or ResourceSearchFilters()
        page_pattern = (
            f"%,{active.page_number},%" if active.page_number is not None else None
        )
        with self.connect() as conn:
            best_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
            for form_index, fts_query in enumerate(forms):
                try:
                    rows = conn.execute(
                        """
                        SELECT pot_id, doc_slug, section_slug, seq, bm25(chunks_fts) AS score
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ? AND pot_id = ?
                          AND (? IS NULL OR doc_slug = ?)
                          AND (? IS NULL OR source_kind = ?)
                          AND (? IS NULL OR page_numbers LIKE ?)
                        ORDER BY score
                        LIMIT ?
                        """,
                        (
                            fts_query,
                            pot_id,
                            active.doc_slug,
                            active.doc_slug,
                            active.source_kind,
                            active.source_kind,
                            page_pattern,
                            page_pattern,
                            limit,
                        ),
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                for row in rows:
                    result = dict(row)
                    key = (
                        str(result["doc_slug"]),
                        str(result["section_slug"]),
                        int(result["seq"]),
                    )
                    current = best_rows.get(key)
                    if current is None:
                        result["query_form"] = form_index
                        best_rows[key] = result
                        continue
                    current_score = current.get("score")
                    result_score = result.get("score")
                    if (
                        isinstance(result_score, (int, float))
                        and isinstance(current_score, (int, float))
                        and result_score < current_score
                    ):
                        result["query_form"] = form_index
                        best_rows[key] = result
            if _is_title_query(query):
                title_rows = conn.execute(
                    """
                    SELECT c.pot_id, c.doc_slug, c.section_slug, c.seq,
                           -1000000.0 + COALESCE(s.ordinal, 0) AS score
                    FROM chunk_text c
                    LEFT JOIN sections s
                      ON s.pot_id = c.pot_id
                     AND s.doc_slug = c.doc_slug
                     AND s.section_slug = c.section_slug
                    WHERE c.pot_id = ? AND c.section_title <> ''
                      AND (? IS NULL OR c.doc_slug = ?)
                      AND (? IS NULL OR c.source_kind = ?)
                      AND (? IS NULL OR c.page_numbers LIKE ?)
                    ORDER BY COALESCE(s.ordinal, 0), c.seq
                    LIMIT ?
                    """,
                    (
                        pot_id,
                        active.doc_slug,
                        active.doc_slug,
                        active.source_kind,
                        active.source_kind,
                        page_pattern,
                        page_pattern,
                        limit,
                    ),
                ).fetchall()
                for row in title_rows:
                    result = dict(row)
                    key = (
                        str(result["doc_slug"]),
                        str(result["section_slug"]),
                        int(result["seq"]),
                    )
                    result["query_form"] = len(forms)
                    best_rows.setdefault(key, result)
            ordered = sorted(
                best_rows.values(),
                key=lambda row: (
                    row.get("score")
                    if isinstance(row.get("score"), (int, float))
                    else float("inf"),
                    row.get("query_form", 0),
                    row["doc_slug"],
                    row["section_slug"],
                    row["seq"],
                ),
            )
            return ordered[:limit]
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

    def clear_chunk_text(self, pot_id: str, doc_slug: str) -> None:
        """Drop the index plane (chunk_text + FTS) for one document."""
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunks_fts WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM chunk_embeddings WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM chunk_text WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.commit()

    def remove_document(self, pot_id: str, doc_slug: str) -> None:
        self.clear_chunk_text(pot_id, doc_slug)
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunk_provenance WHERE pot_id = ? AND doc_slug = ?",
                (pot_id, doc_slug),
            )
            conn.execute(
                "DELETE FROM document_elements WHERE pot_id = ? AND doc_slug = ?",
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
