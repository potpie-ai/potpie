"""``sqlite_fts`` — the stdlib-only lexical profile, and the schema every
profile shares.

SQLite rather than a server-backed engine for one deployment reason: the hosted
service image is single-process by design so uvicorn receives ``SIGTERM``
directly, and Qdrant/Weaviate/Elastic would each need a second container or a
supervisor. FTS5 ships with CPython's ``sqlite3``; the vector arm
(``sqlite_hybrid``) is one loadable extension on top of *this* schema, not a
different one — which is why the table definitions live here and the hybrid
profile subclasses rather than forks.

Three decisions worth reading before changing anything:

**The index is derived, never a source of truth.** Files are the truth. Every
row here is recomputable, so there is no backup, no export, and no
schema-version negotiation: a corrupt or stale index is fixed by
``potpie resource index rebuild --confirm``.

**Pending work is a row state.** A window lands with ``embedding IS NULL`` and
a single writer drains it. That makes "what is outstanding" a ``SELECT``, a
crash a no-op, and the whole background story free of a broker or a scheduler.
``sqlite_fts`` declares no semantic capability, so for it there is never
anything pending — the machinery is here because it belongs to the schema.

**FTS5 ``MATCH`` syntax is not user input.** A query containing ``:``, ``"`` or
``NEAR`` is a syntax error, not zero results, so :func:`fts_match_expression`
tokenizes and quotes before the query ever reaches SQLite. ``unicode61`` is
configured with ``tokenchars '_'`` so ``ERR_QUOTA_EXCEEDED`` and ``get_user_id``
stay single tokens — the identifiers that are the strongest lexical signal and
the weakest embedding one.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from potpie_context_core.ports.resource_index import (
    DEFAULT_DRAIN_BUDGET,
    EMBED_WINDOW_CHARS,
    EMBED_WINDOW_OVERLAP_CHARS,
    MATCH_MODE_DISABLED,
    MATCH_MODE_LEXICAL,
    RRF_K,
    SNIPPET_TARGET_CHARS,
    ChunkHit,
    DrainReport,
    IndexCapabilities,
    IndexReport,
    IndexSearchResult,
    ResourceIndexStatus,
)
from potpie_context_core.ports.resource_store import (
    Chunk,
    DocumentManifest,
    require_resource_slug,
)
from potpie_context_core.resource_to_semantic import document_key, section_key
from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.adapters.outbound.resources.index._unimplemented import (
    UnimplementedSemanticArm,
)

logger = logging.getLogger(__name__)

#: One database for every pot and every document. Pot scoping is a column, the
#: way it is everywhere else in the system, rather than a file per pot: a search
#: is always ``WHERE pot_id = ?`` and never opens a second handle, so the
#: hardest invariant is enforced in one place instead of by filesystem layout.
INDEX_FILENAME = "resources.sqlite3"

#: Windows of a chunk (embedded in slices) versus a section's agent-authored
#: summary (embedded whole). Sections give paraphrase recall at ~25x fewer
#: vectors; chunk windows give passage precision. Both resolve to a chunk id.
WINDOW_KIND_CHUNK = "chunk"
WINDOW_KIND_SECTION = "section"

#: How many candidates each retrieval arm returns before fusion. Wider than the
#: caller's ``limit`` because reciprocal-rank fusion can only reorder what it is
#: given: a chunk ranked 30th lexically and 2nd semantically is exactly the hit
#: hybrid retrieval exists to surface, and a pool cut at ``limit`` never sees it.
ARM_CANDIDATE_MULTIPLIER = 5
ARM_CANDIDATE_MINIMUM = 50

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS meta(
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents(
        pot_id      TEXT    NOT NULL,
        doc         TEXT    NOT NULL,
        revision    INTEGER NOT NULL,
        source_ref  TEXT,
        source_kind TEXT,
        sections    INTEGER NOT NULL DEFAULT 0,
        indexed_at  REAL    NOT NULL,
        PRIMARY KEY (pot_id, doc)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks(
        id            INTEGER PRIMARY KEY,
        pot_id        TEXT    NOT NULL,
        doc           TEXT    NOT NULL,
        section       TEXT    NOT NULL,
        seq           INTEGER NOT NULL,
        resource_id   TEXT    NOT NULL,
        section_title TEXT    NOT NULL DEFAULT '',
        label         TEXT    NOT NULL DEFAULT '',
        text          TEXT    NOT NULL,
        chars         INTEGER NOT NULL,
        revision      INTEGER NOT NULL,
        source_ref    TEXT
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS chunks_resource ON chunks(pot_id, resource_id)",
    "CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(pot_id, doc)",
    # External content: the postings reference ``chunks`` by rowid rather than
    # keeping a second copy of every chunk body. That is ~200MB saved on a
    # 50k-chunk corpus, at the cost of deletes having to hand FTS5 the original
    # column values — which :meth:`_delete_document_rows` does, because it has
    # read them already.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        label,
        section_title,
        content='chunks',
        content_rowid='id',
        tokenize="unicode61 tokenchars '_'"
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS windows(
        id         INTEGER PRIMARY KEY,
        pot_id     TEXT    NOT NULL,
        doc        TEXT    NOT NULL,
        section    TEXT    NOT NULL,
        chunk_id   INTEGER NOT NULL,
        kind       TEXT    NOT NULL,
        ordinal    INTEGER NOT NULL,
        start      INTEGER NOT NULL DEFAULT 0,
        length     INTEGER NOT NULL DEFAULT 0,
        text       TEXT,
        embedding  BLOB,
        embedder   TEXT,
        dimensions INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS windows_chunk ON windows(chunk_id)",
    "CREATE INDEX IF NOT EXISTS windows_doc ON windows(pot_id, doc)",
    # Partial index on exactly the drain's predicate: pending work is a
    # ``SELECT``, and this is what keeps that ``SELECT`` from scanning every
    # window in the corpus once the backlog is empty.
    "CREATE INDEX IF NOT EXISTS windows_pending ON windows(pot_id) WHERE embedding IS NULL",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Function words, dropped from the lexical arm's MATCH expression.
#
# Not an optimization — a correctness fix, measured. Because tokens are OR-ed,
# one "how" or "we" in the query matches nearly every chunk in the corpus, so a
# natural-language question retrieves the whole document set and BM25 then ranks
# noise against noise. On the 19-question answer key that cost more than half of
# top-1: "why did we reject CRDTs" returned the incident postmortem because
# "why/did/we" hit everything, while the one chunk containing "CRDTs" ranked
# fifth.
#
# Kept deliberately small and closed-class: articles, pronouns, prepositions,
# auxiliaries, and the question words. Nothing here can be the answer to
# anything. Domain words a user might plausibly search for ("no", "not", "off",
# "all") are NOT here — an operator searching "not cumulative" means it.
_STOPWORDS = frozenset(
    """
    a an the this that these those
    i we you he she it they them us me my our your its their
    is am are was were be been being do does did doing done
    have has had having will would shall should can could may might must
    of in on at to from by for with without into onto over under about as
    and or but if then than so because while during before after
    what when where which who whom whose why how
    there here any some each every
    """.split()
)


@dataclass(frozen=True, slots=True)
class ArmMatch:
    """One arm's verdict on one chunk: what to quote, and how sure it is.

    ``similarity`` is populated only by an arm that actually measured the query
    against the text. BM25 does not produce one — it is an unbounded log-odds,
    not a [0,1] confidence — so the lexical arm leaves it ``None``. What the
    lexical arm contributes instead is its rank plus
    :func:`term_coverage`, which the reader combines into the relevance it
    votes with.
    """

    snippet: str
    similarity: float | None = None


# --- query preparation ------------------------------------------------------


def query_terms(query: str) -> tuple[str, ...]:
    """The query's content terms: what the lexical arm searches *for*.

    One function so the MATCH expression and :func:`term_coverage` can never
    disagree about what counts as a term — a coverage denominator that included
    words the expression never searched would score every hit short.
    """
    tokens = [token for token in _TOKEN_RE.findall(query or "") if token]
    content = [token for token in tokens if token.lower() not in _STOPWORDS]
    return tuple(content or tokens)


#: Shortest term allowed to match on a prefix rather than whole-token. Below
#: this, a prefix is mostly noise — "up" would cover "upstream", "id" would
#: cover "identity". At four the shortest real casualties are words like "fail",
#: which is exactly the case prefixes exist to catch.
_PREFIX_MIN_CHARS = 4


def term_coverage(text: str, terms: Sequence[str]) -> float | None:
    """Fraction of ``terms`` present in ``text``, tokenized the same way.

    Whole-token first, so ``cat`` is never "covered" by ``catastrophic``. But
    strict token equality alone was measurably too brittle for prose: "when is
    it unsafe to fail over the database" scores the runbook's own failover
    section at 1/3, because the text writes ``failover`` as one word and
    ``fail`` does not equal it. That single miss dropped a correct top-1 answer
    out of the result set entirely.

    So a term of at least :data:`_PREFIX_MIN_CHARS` also counts when it is a
    prefix of a chunk token or vice versa — ``fail``/``failover``,
    ``credit``/``credits``, ``deploy``/``deployment``.

    Deliberately not stemming: no dictionary, no language assumption, nothing
    that can surprise an operator reading the number back. The price is stated
    plainly because it is real — a prefix cannot cross a spelling change, so
    ``retry``/``retries`` and ``terminate``/``termination`` do **not** count,
    and the semantic arm is what carries those pairs. Coverage is a discount on
    over-confidence, not a recall mechanism, so under-counting costs a hit some
    score and never its place in the result set.
    """
    if not terms:
        return None
    present = {token.lower() for token in _TOKEN_RE.findall(text or "")}
    return sum(1 for term in terms if _covers(present, term.lower())) / len(terms)


def _covers(present: frozenset[str] | set[str], term: str) -> bool:
    if term in present:
        return True
    if len(term) < _PREFIX_MIN_CHARS:
        return False
    return any(
        token.startswith(term)
        or (len(token) >= _PREFIX_MIN_CHARS and term.startswith(token))
        for token in present
    )


def fts_match_expression(query: str) -> str | None:
    """Turn free text into a legal FTS5 ``MATCH`` expression, or ``None``.

    Every token is quoted, which does two jobs at once: it neutralizes the
    operators a user phrase contains by accident (``:``, ``*``, ``NEAR``,
    ``AND``) and it stops a stray quote from turning a search into a syntax
    error the caller sees as a crash rather than as zero results.

    Tokens are ``OR``-ed rather than ``AND``-ed, which buys the recall ``AND``
    throws away the moment one word of a five-word question is absent from an
    otherwise perfect passage. The price is that every OR-ed term widens the
    candidate set, so :data:`_STOPWORDS` comes out first — see there for the
    measurement that made it non-optional.

    A query made *entirely* of stopwords keeps them. "how do I" retrieving
    loosely is a poor answer; retrieving nothing is a wrong one, and the
    semantic arm is the better judge of that query anyway.
    """
    tokens = list(query_terms(query))
    if not tokens:
        return None
    # Bound the expression: a pasted paragraph is a legal query and a
    # thousand-term OR is a pathological one.
    return " OR ".join(f'"{token}"' for token in tokens[:64])


def embed_windows(text: str) -> tuple[tuple[int, int], ...]:
    """Slice a chunk into ``(start, length)`` sub-windows for embedding.

    Whole-chunk embedding is not an option: MiniLM-L6-v2 truncates at 256
    tokens and a 4,000-char chunk is ~792, so embedding the chunk would index
    its first third and silently discard the rest. Windows overlap so a fact
    straddling a boundary is whole in at least one of them.
    """
    body = text or ""
    if not body.strip():
        return ()
    if len(body) <= EMBED_WINDOW_CHARS:
        return ((0, len(body)),)
    stride = max(EMBED_WINDOW_CHARS - EMBED_WINDOW_OVERLAP_CHARS, 1)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(body):
        spans.append((start, min(EMBED_WINDOW_CHARS, len(body) - start)))
        if start + EMBED_WINDOW_CHARS >= len(body):
            break
        start += stride
    return tuple(spans)


def reciprocal_rank_fusion(
    *arms: Sequence[int], k: int = RRF_K
) -> dict[int, tuple[float, dict[int, int]]]:
    """Fuse ranked id lists into ``id -> (score, {arm_index: rank})``.

    Rank-based rather than score-based on purpose: BM25 is an unbounded
    negative log-odds and cosine similarity is a bounded [0,1], so no
    normalization of the two is defensible across corpora. Their *orderings*
    are comparable, which is the whole argument for RRF.
    """
    fused: dict[int, tuple[float, dict[int, int]]] = {}
    for arm_index, ids in enumerate(arms):
        for position, row_id in enumerate(ids):
            rank = position + 1
            score, ranks = fused.get(row_id, (0.0, {}))
            ranks[arm_index] = rank
            fused[row_id] = (score + 1.0 / (k + rank), ranks)
    return fused


# --- the index --------------------------------------------------------------


@dataclass(slots=True)
class SqliteFtsResourceIndex:
    """Lexical retrieval over stored chunks, plus the shared schema.

    One connection, opened lazily and guarded by a lock. Lazily because
    :meth:`status` must answer on a machine where the database cannot be
    created at all, and constructing the object is not the moment to find that
    out. Guarded because the drain runs on its own thread while RPC handlers
    read on another, and SQLite's own thread check would reject the second one.
    """

    home: Path = field(default_factory=default_home)
    profile: str = "sqlite_fts"
    embedder: Any = None
    """Unused by this profile; accepted so both profiles share one factory."""

    shared_store: bool = False
    """Set when the *file* store underneath is a shared volume.

    The index is process-local, so on a shared store a document imported
    through one replica is invisible to another until that replica rebuilds.
    Nothing here can fix that; the flag exists so ``status`` can say it rather
    than let it surface as a document that is findable half the time."""

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _conn: sqlite3.Connection | None = field(default=None, repr=False)
    _open_error: str | None = field(default=None, repr=False)

    # --- capabilities -------------------------------------------------------

    def capabilities(self) -> IndexCapabilities:
        return IndexCapabilities(
            profile=self.profile,
            lexical=True,
            snippets=True,
            incremental=True,
        )

    @property
    def semantic_arm(self) -> Any:
        """The vector half — fail-closed here, real in ``sqlite_hybrid``."""
        return UnimplementedSemanticArm(self.profile)

    # --- connection ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.home / "index" / INDEX_FILENAME

    def _connect(self) -> sqlite3.Connection:
        """Open (and migrate) the database, or raise the reason it cannot be."""
        with self._lock:
            if self._conn is not None:
                return self._conn
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self.path,
                # The drain thread and the request thread share this handle;
                # every use is already inside ``self._lock``.
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            # WAL so a reader is never blocked by the drain's writes — the
            # single behavioural reason the drain can be a plain thread rather
            # than something that has to coordinate with request handling.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                for statement in _SCHEMA:
                    conn.execute(statement)
                conn.commit()
            except sqlite3.OperationalError:
                conn.close()
                raise
            self._conn = conn
            self._reconcile_embedder(conn)
            return conn

    def _try_connect(self) -> sqlite3.Connection | None:
        """Connect, remembering the failure instead of raising.

        Used by the paths that must degrade rather than fail: ``status`` is a
        diagnostic and ``search`` has an envelope to fill, so neither may turn a
        missing FTS5 build or an unwritable home into an exception."""
        try:
            return self._connect()
        except (sqlite3.Error, OSError) as exc:
            self._open_error = str(exc)
            logger.debug("resource index unavailable at %s: %s", self.path, exc)
            return None

    def _reconcile_embedder(self, conn: sqlite3.Connection) -> None:
        """Invalidate vectors when the embedder or its dimensions changed.

        Marking stale rows pending is the *entire* migration story (a non-goal
        spells this out): vectors from two models do not share a space, so the
        only honest options are re-embed or refuse to answer, and the drain
        already knows how to re-embed. ``sqlite_fts`` has no embedder, so this
        is a no-op for it — but the reconciliation lives here because the rows
        it guards are part of this schema.
        """
        name, dimensions = self._embedder_identity()
        if name is None:
            return
        stored_name = self._meta_get(conn, "embedder")
        stored_dims = self._meta_get(conn, "dimensions")
        if stored_name == name and stored_dims == str(dimensions):
            return
        if stored_name is not None:
            logger.info(
                "resource index embedder changed (%s/%s -> %s/%s); "
                "marking every vector pending",
                stored_name,
                stored_dims,
                name,
                dimensions,
            )
            conn.execute(
                "UPDATE windows SET embedding=NULL, embedder=NULL, dimensions=NULL"
            )
        self._meta_set(conn, "embedder", name)
        self._meta_set(conn, "dimensions", str(dimensions))
        conn.commit()

    def _embedder_identity(self) -> tuple[str | None, int | None]:
        if self.embedder is None:
            return (None, None)
        try:
            return (str(self.embedder.name), int(self.embedder.dimensions))
        except Exception:  # noqa: BLE001 - a broken embedder must not break open()
            logger.warning("resource index embedder did not report its identity")
            return (None, None)

    @staticmethod
    def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    @staticmethod
    def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --- writes -------------------------------------------------------------

    def index_document(
        self,
        *,
        pot_id: str,
        manifest: DocumentManifest,
        chunks: tuple[Chunk, ...],
    ) -> IndexReport:
        """Replace one document's rows with this revision's.

        Replace rather than merge: a revision can drop a section, renumber a
        chunk, or rewrite a body under an unchanged id, and a merge would leave
        the index answering with text the store no longer holds. The whole
        document is one transaction, so a failure leaves the prior revision
        searchable instead of a half-updated one.
        """
        doc = require_resource_slug(manifest.doc, kind="document")
        conn = self._connect()
        titles = {row.slug: row.title for row in manifest.sections}
        labels = {
            (row.slug, ref.seq): ref.label
            for row in manifest.sections
            for ref in row.chunks
        }
        windows_written = 0
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                removed = self._delete_document_rows(conn, pot_id=pot_id, doc=doc)
                first_chunk_of_section: dict[str, int] = {}
                for chunk in chunks:
                    cursor = conn.execute(
                        "INSERT INTO chunks(pot_id, doc, section, seq, resource_id, "
                        "section_title, label, text, chars, revision, source_ref) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            pot_id,
                            doc,
                            chunk.section,
                            chunk.seq,
                            chunk.resource_id,
                            titles.get(chunk.section, ""),
                            labels.get((chunk.section, chunk.seq), ""),
                            chunk.text,
                            chunk.chars,
                            chunk.revision,
                            chunk.source_ref,
                        ),
                    )
                    chunk_id = int(cursor.lastrowid or 0)
                    conn.execute(
                        "INSERT INTO chunks_fts(rowid, text, label, section_title) "
                        "VALUES(?,?,?,?)",
                        (
                            chunk_id,
                            chunk.text,
                            labels.get((chunk.section, chunk.seq), ""),
                            titles.get(chunk.section, ""),
                        ),
                    )
                    first_chunk_of_section.setdefault(chunk.section, chunk_id)
                    windows_written += self._write_chunk_windows(
                        conn,
                        pot_id=pot_id,
                        doc=doc,
                        section=chunk.section,
                        chunk_id=chunk_id,
                        text=chunk.text,
                    )
                windows_written += self._write_section_windows(
                    conn,
                    pot_id=pot_id,
                    doc=doc,
                    manifest=manifest,
                    anchors=first_chunk_of_section,
                )
                conn.execute(
                    "INSERT INTO documents(pot_id, doc, revision, source_ref, "
                    "source_kind, sections, indexed_at) VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(pot_id, doc) DO UPDATE SET "
                    "revision=excluded.revision, source_ref=excluded.source_ref, "
                    "source_kind=excluded.source_kind, sections=excluded.sections, "
                    "indexed_at=excluded.indexed_at",
                    (
                        pot_id,
                        doc,
                        manifest.revision,
                        manifest.source_ref,
                        manifest.source_kind,
                        len(manifest.sections),
                        time.time(),
                    ),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return IndexReport(
            doc=doc,
            profile=self.profile,
            sections=len(manifest.sections),
            chunks=len(chunks),
            windows=windows_written,
            # Only a profile that embeds leaves work behind. Reporting the
            # window count here would tell a lexical-only caller to wait for a
            # drain that will never run.
            pending_embeddings=windows_written if self.capabilities().semantic else 0,
            removed_chunks=removed,
        )

    def _write_chunk_windows(
        self,
        conn: sqlite3.Connection,
        *,
        pot_id: str,
        doc: str,
        section: str,
        chunk_id: int,
        text: str,
    ) -> int:
        spans = embed_windows(text)
        conn.executemany(
            "INSERT INTO windows(pot_id, doc, section, chunk_id, kind, ordinal, "
            "start, length, text) VALUES(?,?,?,?,?,?,?,?,NULL)",
            [
                (
                    pot_id,
                    doc,
                    section,
                    chunk_id,
                    WINDOW_KIND_CHUNK,
                    ordinal,
                    start,
                    length,
                )
                for ordinal, (start, length) in enumerate(spans)
            ],
        )
        return len(spans)

    def _write_section_windows(
        self,
        conn: sqlite3.Connection,
        *,
        pot_id: str,
        doc: str,
        manifest: DocumentManifest,
        anchors: Mapping[str, int],
    ) -> int:
        """Embed each section's summary, anchored to its first chunk.

        The summaries already exist and there are ~25x fewer of them than
        chunk windows, so they are the cheapest recall available: sections
        answer a paraphrase, chunk windows answer a phrase. A section hit
        resolves to the section's first chunk because search lands on a section
        and the agent then picks a chunk by label — that is the documented flow,
        and it keeps the promise that every hit is addressable by one id.
        """
        rows = [
            (
                pot_id,
                doc,
                row.slug,
                anchors[row.slug],
                WINDOW_KIND_SECTION,
                0,
                0,
                0,
                f"{row.title}\n{row.summary}".strip(),
            )
            for row in manifest.sections
            if row.slug in anchors and (row.summary.strip() or row.title.strip())
        ]
        conn.executemany(
            "INSERT INTO windows(pot_id, doc, section, chunk_id, kind, ordinal, "
            "start, length, text) VALUES(?,?,?,?,?,?,?,?,?)",
            rows,
        )
        return len(rows)

    def _delete_document_rows(
        self, conn: sqlite3.Connection, *, pot_id: str, doc: str
    ) -> int:
        """Drop a document's chunks, windows, and postings. Returns chunks removed.

        The FTS5 ``'delete'`` command needs the *original* column values, not
        just the rowid — external content means the postings cannot look them
        up once the row is gone. So the rows are read first and the postings
        are retired before the base table is touched.
        """
        rows = conn.execute(
            "SELECT id, text, label, section_title FROM chunks WHERE pot_id=? AND doc=?",
            (pot_id, doc),
        ).fetchall()
        if rows:
            conn.executemany(
                "INSERT INTO chunks_fts(chunks_fts, rowid, text, label, section_title) "
                "VALUES('delete', ?, ?, ?, ?)",
                [(r["id"], r["text"], r["label"], r["section_title"]) for r in rows],
            )
            conn.executemany(
                "DELETE FROM windows WHERE chunk_id=?", [(r["id"],) for r in rows]
            )
        conn.execute("DELETE FROM chunks WHERE pot_id=? AND doc=?", (pot_id, doc))
        conn.execute("DELETE FROM documents WHERE pot_id=? AND doc=?", (pot_id, doc))
        return len(rows)

    def drop_document(self, *, pot_id: str, slug: str) -> bool:
        doc = require_resource_slug(slug, kind="document")
        conn = self._try_connect()
        if conn is None:
            return False
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                removed = self._delete_document_rows(conn, pot_id=pot_id, doc=doc)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return removed > 0

    def purge_pot(self, pot_id: str) -> bool:
        conn = self._try_connect()
        if conn is None:
            return False
        with self._lock:
            docs = [
                row["doc"]
                for row in conn.execute(
                    "SELECT doc FROM documents WHERE pot_id=?", (pot_id,)
                ).fetchall()
            ]
            # Orphan chunks (a document row lost to a partial write) would
            # survive a documents-driven purge, and pot isolation is the one
            # invariant that cannot be left to tidiness.
            orphans = [
                row["doc"]
                for row in conn.execute(
                    "SELECT DISTINCT doc FROM chunks WHERE pot_id=?", (pot_id,)
                ).fetchall()
            ]
            try:
                conn.execute("BEGIN IMMEDIATE")
                removed = 0
                for doc in dict.fromkeys([*docs, *orphans]):
                    removed += self._delete_document_rows(conn, pot_id=pot_id, doc=doc)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return bool(docs or orphans)

    # --- reads --------------------------------------------------------------

    def search(
        self,
        *,
        pot_id: str,
        query: str,
        limit: int = 12,
        doc: str | None = None,
    ) -> IndexSearchResult:
        """Run every declared arm, fuse by reciprocal rank, return one item per chunk."""
        caps = self.capabilities()
        conn = self._try_connect()
        if conn is None:
            return IndexSearchResult(
                profile=self.profile,
                match_mode=MATCH_MODE_DISABLED,
                detail=self._open_error or "resource index is unavailable",
            )
        pool = max(limit * ARM_CANDIDATE_MULTIPLIER, ARM_CANDIDATE_MINIMUM)
        lexical = self._lexical_arm(
            conn, pot_id=pot_id, query=query, limit=pool, doc=doc
        )
        semantic = (
            self._semantic_arm(conn, pot_id=pot_id, query=query, limit=pool, doc=doc)
            if caps.semantic
            else {}
        )
        if not lexical and not semantic:
            return IndexSearchResult(
                profile=self.profile,
                match_mode=self._answered_mode(caps, lexical, semantic),
                lexical_candidates=0,
                semantic_candidates=0,
            )
        fused = reciprocal_rank_fusion(list(lexical), list(semantic))
        ordered = sorted(
            fused.items(),
            # Fused score first; the chunk id breaks ties so two runs over the
            # same corpus return the same order.
            key=lambda item: (-item[1][0], item[0]),
        )[:limit]
        hits = self._hydrate(
            conn,
            ordered=ordered,
            lexical=lexical,
            semantic=semantic,
            caps=caps,
            terms=query_terms(query),
        )
        return IndexSearchResult(
            profile=self.profile,
            match_mode=self._answered_mode(caps, lexical, semantic),
            hits=hits,
            lexical_candidates=len(lexical),
            semantic_candidates=len(semantic),
        )

    @staticmethod
    def _answered_mode(
        caps: IndexCapabilities,
        lexical: Mapping[int, ArmMatch],
        semantic: Mapping[int, ArmMatch],
    ) -> str:
        """Name the arms that actually answered, not the ones declared.

        A hybrid profile whose vectors are still draining answers lexically,
        and saying ``"hybrid"`` there would misreport the retrieval an agent is
        about to trust. Declared capability is what ``status`` reports; this is
        what *this* search did.
        """
        if lexical and semantic:
            return caps.match_mode()
        if semantic:
            return "semantic"
        if lexical:
            return MATCH_MODE_LEXICAL
        return caps.match_mode()

    def _lexical_arm(
        self,
        conn: sqlite3.Connection,
        *,
        pot_id: str,
        query: str,
        limit: int,
        doc: str | None,
    ) -> dict[int, ArmMatch]:
        """BM25 over chunk text, in rank order. Insertion order *is* the rank."""
        expression = fts_match_expression(query)
        if expression is None:
            return {}
        sql = (
            "SELECT c.id AS id, "
            "snippet(chunks_fts, 0, '', '', '…', 16) AS snip "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND c.pot_id = ?"
        )
        params: list[Any] = [expression, pot_id]
        if doc:
            sql += " AND c.doc = ?"
            params.append(doc)
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(limit)
        with self._lock:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                # A malformed MATCH is a defect in :func:`fts_match_expression`,
                # not the caller's problem, and a search that raises would fail
                # a whole envelope. Log it and answer with the other arm.
                logger.warning("resource index lexical arm failed: %s", exc)
                return {}
        return {
            int(row["id"]): ArmMatch(snippet=str(row["snip"] or "")) for row in rows
        }

    def _semantic_arm(
        self,
        conn: sqlite3.Connection,
        *,
        pot_id: str,
        query: str,
        limit: int,
        doc: str | None,
    ) -> dict[int, ArmMatch]:
        """Vector arm — empty here; ``sqlite_hybrid`` overrides it."""
        del conn, pot_id, query, limit, doc
        return {}

    def _hydrate(
        self,
        conn: sqlite3.Connection,
        *,
        ordered: Sequence[tuple[int, tuple[float, dict[int, int]]]],
        lexical: Mapping[int, ArmMatch],
        semantic: Mapping[int, ArmMatch],
        caps: IndexCapabilities,
        terms: Sequence[str] = (),
    ) -> tuple[ChunkHit, ...]:
        """Turn fused chunk ids into hits, in one query rather than one per hit."""
        if not ordered:
            return ()
        ids = [row_id for row_id, _ in ordered]
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = {
                int(row["id"]): row
                for row in conn.execute(
                    f"SELECT id, doc, section, seq, resource_id, section_title, label, "
                    f"text, chars, revision, source_ref FROM chunks "
                    f"WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
            }
        hits: list[ChunkHit] = []
        for rank, (row_id, (score, arm_ranks)) in enumerate(ordered, start=1):
            row = rows.get(row_id)
            if row is None:
                # The document was dropped between the arms and the hydrate.
                # Skipping is right: the chunk id no longer resolves, so
                # returning it would hand an agent a dead ``resource get``.
                continue
            lexical_match = lexical.get(row_id)
            semantic_match = semantic.get(row_id)
            # The lexical snippet wins when both arms matched: it is centred on
            # the query's own terms, where the semantic one is centred on a
            # window boundary and may not contain them at all.
            quoted = (
                (lexical_match.snippet if lexical_match else "")
                or (semantic_match.snippet if semantic_match else "")
                or str(row["text"] or "")
            )
            hits.append(
                ChunkHit(
                    resource_id=str(row["resource_id"]),
                    doc=str(row["doc"]),
                    section=str(row["section"]),
                    seq=int(row["seq"]),
                    # Pure string manipulation over slugs that cannot contain
                    # ``:`` or ``/`` — which is what makes the graph tie free.
                    document_key=document_key(str(row["doc"])),
                    section_key=section_key(str(row["doc"]), str(row["section"])),
                    section_title=str(row["section_title"] or ""),
                    label=str(row["label"] or ""),
                    snippet=_clean_snippet(quoted),
                    chars=int(row["chars"]),
                    revision=int(row["revision"]),
                    score=round(score, 6),
                    rank=rank,
                    match_mode=_hit_mode(arm_ranks, caps),
                    source_ref=row["source_ref"],
                    lexical_rank=arm_ranks.get(0),
                    semantic_rank=arm_ranks.get(1),
                    similarity=semantic_match.similarity if semantic_match else None,
                    # Measured against the whole chunk, not the snippet: the
                    # snippet is one window chosen around the best match, so
                    # scoring coverage on it would report every hit as perfect.
                    term_coverage=term_coverage(str(row["text"] or ""), terms),
                )
            )
        return tuple(hits)

    # --- drain --------------------------------------------------------------

    def drain(
        self, *, pot_id: str | None = None, budget: int = DEFAULT_DRAIN_BUDGET
    ) -> DrainReport:
        """No-op for a profile with no vectors: nothing is ever pending."""
        del pot_id, budget
        return DrainReport(profile=self.profile)

    def pending_count(self, *, pot_id: str | None = None) -> int:
        conn = self._try_connect()
        if conn is None or not self.capabilities().semantic:
            return 0
        sql = "SELECT COUNT(*) AS n FROM windows WHERE embedding IS NULL"
        params: tuple[Any, ...] = ()
        if pot_id:
            sql += " AND pot_id = ?"
            params = (pot_id,)
        with self._lock:
            return int(conn.execute(sql, params).fetchone()["n"])

    # --- diagnostics --------------------------------------------------------

    def status(self, *, pot_id: str | None = None) -> ResourceIndexStatus:
        """Report readiness and counts. Never raises — see the port docstring."""
        caps = self.capabilities()
        name, dimensions = self._embedder_identity()
        base = ResourceIndexStatus(
            profile=self.profile,
            ready=False,
            capabilities=caps.implemented(),
            match_mode=caps.match_mode(),
            embedder=name,
            dimensions=dimensions,
            location=str(self.path),
            replica=replica_identity(),
            shared_store=self.shared_store,
        )
        conn = self._try_connect()
        if conn is None:
            return _with(base, detail=self._open_error or "index database unavailable")
        scope = "WHERE pot_id = ?" if pot_id else ""
        params: tuple[Any, ...] = (pot_id,) if pot_id else ()
        try:
            with self._lock:
                documents = int(
                    conn.execute(
                        f"SELECT COUNT(*) AS n FROM documents {scope}", params
                    ).fetchone()["n"]
                )
                chunks = int(
                    conn.execute(
                        f"SELECT COUNT(*) AS n FROM chunks {scope}", params
                    ).fetchone()["n"]
                )
                windows = int(
                    conn.execute(
                        f"SELECT COUNT(*) AS n FROM windows {scope}", params
                    ).fetchone()["n"]
                )
        except sqlite3.Error as exc:
            return _with(base, detail=str(exc))
        return _with(
            base,
            ready=True,
            documents=documents,
            chunks=chunks,
            windows=windows,
            pending_embeddings=self.pending_count(pot_id=pot_id),
            detail=_staleness_detail(self.shared_store),
        )

    def rebuild_postings(self) -> None:
        """Rebuild the FTS index from ``chunks``.

        The repair for the one way external content can drift: a ``'delete'``
        whose column values did not match the originals leaves postings behind.
        Cheap, and the reason ``resource index rebuild`` can promise identical
        results rather than merely fresh ones."""
        conn = self._connect()
        with self._lock:
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            conn.commit()

    def documents(self, *, pot_id: str) -> tuple[str, ...]:
        """Which documents this index holds for a pot, for ``rebuild``."""
        conn = self._try_connect()
        if conn is None:
            return ()
        with self._lock:
            return tuple(
                str(row["doc"])
                for row in conn.execute(
                    "SELECT doc FROM documents WHERE pot_id=? ORDER BY doc", (pot_id,)
                ).fetchall()
            )


# --- helpers ----------------------------------------------------------------


def replica_identity() -> str:
    """Which process answered, for the shared-store honesty note in ``status``."""
    import socket

    return f"{socket.gethostname()}:{os.getpid()}"


def _staleness_detail(shared_store: bool) -> str | None:
    if not shared_store:
        return None
    return (
        "the document store is shared but this index is process-local: a document "
        "imported through another replica is invisible here until "
        "'potpie resource index rebuild --confirm' runs on this one. Run a single "
        "replica, or rebuild on deploy."
    )


def _hit_mode(arm_ranks: Mapping[int, int], caps: IndexCapabilities) -> str:
    """How *this* hit was found, which is not always how the profile can search."""
    lexical = 0 in arm_ranks
    semantic = 1 in arm_ranks
    if lexical and semantic:
        return caps.match_mode()
    if semantic:
        return "semantic"
    if lexical:
        return MATCH_MODE_LEXICAL
    return MATCH_MODE_DISABLED


def _clean_snippet(text: str) -> str:
    """Collapse a matched window into one line of at most ~240 chars.

    Never a chunk body: twelve 4,000-char chunks is ~48k chars of envelope,
    which is exactly the budget the two-call search→``get`` path exists to
    protect."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= SNIPPET_TARGET_CHARS:
        return collapsed
    return collapsed[: SNIPPET_TARGET_CHARS - 1].rstrip() + "…"


def _with(status: ResourceIndexStatus, **changes: Any) -> ResourceIndexStatus:
    import dataclasses

    return dataclasses.replace(status, **changes)


def iter_window_texts(
    rows: Iterable[sqlite3.Row], *, bodies: Mapping[int, str]
) -> list[str]:
    """Materialize each pending window's text for the embedder.

    Chunk windows are stored as offsets into ``chunks.text`` rather than as
    copies — a 50k-chunk corpus would otherwise carry a third full copy of
    every body — so the slice happens here, at the one moment the text is
    actually needed.
    """
    texts: list[str] = []
    for row in rows:
        if row["kind"] == WINDOW_KIND_SECTION:
            texts.append(str(row["text"] or ""))
            continue
        body = bodies.get(int(row["chunk_id"]), "")
        start = int(row["start"])
        texts.append(body[start : start + int(row["length"])])
    return texts


__all__ = [
    "ARM_CANDIDATE_MINIMUM",
    "ARM_CANDIDATE_MULTIPLIER",
    "INDEX_FILENAME",
    "ArmMatch",
    "WINDOW_KIND_CHUNK",
    "WINDOW_KIND_SECTION",
    "SqliteFtsResourceIndex",
    "embed_windows",
    "fts_match_expression",
    "iter_window_texts",
    "reciprocal_rank_fusion",
    "replica_identity",
]
