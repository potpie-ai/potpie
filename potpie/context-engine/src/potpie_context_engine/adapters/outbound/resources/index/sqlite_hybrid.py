"""``sqlite_hybrid`` — the default profile: BM25 and vectors over one schema.

Everything structural comes from :class:`SqliteFtsResourceIndex`; this adds the
second retrieval arm and the drain that fills it. Two things are worth being
explicit about.

**Degradation is reported, not hidden.** ``sqlite-vec`` is a loadable
extension, and some CPython builds ship ``sqlite3`` with
``enable_load_extension`` compiled out. When the extension will not load — or
no embedder is wired — the profile keeps its *name* and drops its
*capabilities* to lexical, with the reason in ``status().detail``. That is the
whole point of a capability bundle: a caller sees "you asked for hybrid, you
are getting lexical, here is why" instead of silently better-or-worse recall.

**Distance is computed, not indexed.** Vectors live as ``float32`` blobs in
``windows`` and ``vec_distance_cosine`` scores them in C. No ``vec0`` virtual
table: that would bake the embedder's dimension into the schema, so a model
swap would become a migration — and a migration is exactly what a *derived*
index must never need. A brute-force scan over one pot's windows is a
vectorized pass over a few tens of MB; if a corpus ever outgrows that, an ANN
index is a change behind this method and nothing above it moves.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import sqlite3
import time
from array import array
from dataclasses import dataclass, field
from typing import Any, Sequence

from potpie_context_core.ports.resource_index import (
    DEFAULT_DRAIN_BUDGET,
    DrainReport,
    IndexCapabilities,
    ResourceIndexStatus,
)
from potpie_context_engine.adapters.outbound.resources.index.sqlite_fts import (
    ArmMatch,
    SqliteFtsResourceIndex,
    WINDOW_KIND_SECTION,
    _clean_snippet,
    iter_window_texts,
)

logger = logging.getLogger(__name__)

#: One embed call per drain step, so a step stays interruptible. The daemon
#: hard-kills 10s after SIGTERM with no drain hook, and an unfinished batch
#: simply stays ``NULL`` — the reason this needs no shutdown coordination.
EMBED_BATCH = 64


def load_vector_extension(conn: sqlite3.Connection) -> str | None:
    """Load ``sqlite-vec`` into ``conn``; return the failure reason, or ``None``.

    Failure is a fact to report, not an exception to propagate: this runs on
    the connect path, and an index that raises here would take ``resource
    import`` and ``potpie doctor`` down with it over a capability that is
    supposed to be optional.
    """
    try:
        import sqlite_vec
    except ImportError as exc:  # pragma: no cover - declared dependency
        return f"sqlite-vec is not installed: {exc}"
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        return (
            "this Python's sqlite3 was built without extension loading, so "
            "sqlite-vec cannot be used"
        )
    except sqlite3.Error as exc:
        return f"sqlite3 refused to enable extension loading: {exc}"
    try:
        sqlite_vec.load(conn)
    except Exception as exc:  # noqa: BLE001 - any load failure degrades the same way
        return f"sqlite-vec failed to load: {exc}"
    finally:
        # Narrow the window: nothing else in this process loads extensions, and
        # leaving the door open would let any later SQL on this connection do it.
        with contextlib.suppress(sqlite3.Error, AttributeError):
            conn.enable_load_extension(False)
    return None


def pack_vector(values: Sequence[float]) -> bytes:
    """Serialize an embedding the way ``sqlite-vec`` reads it: raw ``float32``."""
    return array("f", [float(v) for v in values]).tobytes()


#: Sentinel for "the extension has not been probed on this connection yet",
#: distinct from ``None`` which means "probed, and it loaded".
_UNPROBED = "\x00unprobed"


@dataclass(slots=True)
class SqliteHybridResourceIndex(SqliteFtsResourceIndex):
    """BM25 + vector similarity, fused by reciprocal rank."""

    profile: str = "sqlite_hybrid"
    _vec_state: str | None = field(default=_UNPROBED, repr=False)
    """The extension probe's verdict: ``None`` once loaded, else the reason."""

    def capabilities(self) -> IndexCapabilities:
        """Declare what this instance can actually do *right now*.

        Deliberately not a constant. The extension may be unloadable and the
        embedder may be switched off (``CONTEXT_ENGINE_EMBEDDER=none``), and in
        both cases the honest answer is a lexical profile wearing a hybrid
        name — which the search path then reads to decide whether to run the
        second arm at all.
        """
        vectors = self._vectors_ready()
        return IndexCapabilities(
            profile=self.profile,
            lexical=True,
            semantic=vectors,
            hybrid=vectors,
            snippets=True,
            incremental=True,
        )

    def _vectors_ready(self) -> bool:
        if self.embedder is None:
            return False
        return self._vector_error() is None

    def _vector_error(self) -> str | None:
        """Why the vector arm is unavailable, or ``None`` when it is fine."""
        if self.embedder is None:
            return (
                "no embedder is wired (CONTEXT_ENGINE_EMBEDDER=none), so only "
                "lexical retrieval is available"
            )
        conn = self._try_connect()
        if conn is None:
            return self._open_error or "index database unavailable"
        return self._extension_state(conn)

    def _extension_state(self, conn: sqlite3.Connection) -> str | None:
        """Load the extension once per connection and cache the verdict.

        Cached because ``capabilities()`` is called on every search and the
        probe is a dlopen; and because a failure that is logged once is a
        diagnostic, while one logged per query is noise.
        """
        if self._vec_state is not _UNPROBED:
            return self._vec_state
        with self._lock:
            reason = load_vector_extension(conn)
        self._vec_state = reason
        if reason is not None:
            logger.warning("resource index vector arm unavailable: %s", reason)
        return reason

    # --- semantic arm -------------------------------------------------------

    def _semantic_arm(
        self,
        conn: sqlite3.Connection,
        *,
        pot_id: str,
        query: str,
        limit: int,
        doc: str | None,
    ) -> dict[int, ArmMatch]:
        """k-NN over embedded windows, collapsed to one match per chunk.

        Collapsed because the contract is one item per chunk: a chunk with five
        matching windows must take one result slot, not five. The best window
        wins and supplies the snippet, which is what makes a semantic-only hit
        readable — an agent gets the passage that matched rather than the head
        of a 4,000-char body.
        """
        if not (query or "").strip():
            return {}
        vector = self._embed_query(query)
        if vector is None:
            return {}
        sql = (
            "SELECT w.chunk_id AS chunk_id, w.kind AS kind, w.start AS start, "
            "w.length AS length, w.text AS text, "
            "vec_distance_cosine(w.embedding, ?) AS distance "
            "FROM windows w "
            "WHERE w.pot_id = ? AND w.embedding IS NOT NULL"
        )
        params: list[Any] = [vector, pot_id]
        if doc:
            sql += " AND w.doc = ?"
            params.append(doc)
        sql += " ORDER BY distance LIMIT ?"
        # Windows outnumber chunks several-fold, so ask for more rows than the
        # caller's pool or the collapse below returns far fewer chunks than the
        # fusion was sized for.
        params.append(limit * 4)
        with self._lock:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                logger.warning("resource index semantic arm failed: %s", exc)
                return {}
        # Rows arrive nearest-first, so the first appearance of a chunk is its
        # best window: keeping that one and dropping the rest is what enforces
        # "one item per chunk" without a second sort.
        best: dict[int, ArmMatch] = {}
        bodies: dict[int, str] = {}
        for row in rows:
            chunk_id = int(row["chunk_id"])
            if chunk_id in best or len(best) >= limit:
                continue
            if row["distance"] is None:
                # ``vec_distance_cosine`` returns NULL against a zero-norm
                # vector, which a stored window can be when its text carried no
                # features the embedder recognises. Undefined is not "far", and
                # certainly not "identical" — the row is simply not a semantic
                # answer, so it is skipped rather than scored.
                continue
            # The distance is 1 - cos over [-1, 1], so it can exceed 1 for an
            # opposed vector. Clamping keeps the value the ranker sees inside
            # the [0, 1] every other similarity in the system lives in.
            similarity = max(0.0, min(1.0, 1.0 - float(row["distance"])))
            if row["kind"] == WINDOW_KIND_SECTION:
                best[chunk_id] = ArmMatch(
                    snippet=_clean_snippet(str(row["text"] or "")),
                    similarity=similarity,
                )
                continue
            if chunk_id not in bodies:
                body = conn.execute(
                    "SELECT text FROM chunks WHERE id=?", (chunk_id,)
                ).fetchone()
                bodies[chunk_id] = "" if body is None else str(body["text"] or "")
            start = int(row["start"])
            best[chunk_id] = ArmMatch(
                snippet=_clean_snippet(
                    bodies[chunk_id][start : start + int(row["length"])]
                ),
                similarity=similarity,
            )
        return best

    def _embed_query(self, query: str) -> bytes | None:
        if self.embedder is None:
            return None
        try:
            vector = self.embedder.embed(query)
        except Exception as exc:  # noqa: BLE001 - a model failure degrades the read
            logger.warning("resource index could not embed the query: %s", exc)
            return None
        if not any(vector):
            # A query with no features the embedder recognises ("-", "***")
            # embeds to the zero vector, against which cosine distance is
            # undefined for every row. Skipping the arm is the honest answer:
            # there is no semantic question here to compare anything to.
            return None
        return pack_vector(vector)

    # --- drain --------------------------------------------------------------

    def drain(
        self, *, pot_id: str | None = None, budget: int = DEFAULT_DRAIN_BUDGET
    ) -> DrainReport:
        """Embed up to ``budget`` pending windows; report what is left.

        The one step of the loop over derived state. Everything about it is a
        consequence of pending work being a row state rather than a message: it
        is safe to call concurrently (writers serialize on the connection lock
        and a row already filled is simply not selected again), safe to
        interrupt (the unfinished rows stay ``NULL``), and safe to lose (the
        next call resumes).
        """
        started = time.monotonic()
        if not self.capabilities().semantic:
            return DrainReport(profile=self.profile, detail=self._vector_error())
        conn = self._try_connect()
        if conn is None:
            return DrainReport(profile=self.profile, detail=self._open_error)
        embedded = 0
        batches = 0
        while embedded < budget:
            filled = self._drain_batch(
                conn, pot_id=pot_id, size=min(EMBED_BATCH, budget - embedded)
            )
            if filled == 0:
                break
            embedded += filled
            batches += 1
        return DrainReport(
            profile=self.profile,
            embedded=embedded,
            remaining=self.pending_count(pot_id=pot_id),
            batches=batches,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def _drain_batch(
        self, conn: sqlite3.Connection, *, pot_id: str | None, size: int
    ) -> int:
        sql = (
            "SELECT id, chunk_id, kind, start, length, text FROM windows "
            "WHERE embedding IS NULL"
        )
        params: list[Any] = []
        if pot_id:
            sql += " AND pot_id = ?"
            params.append(pot_id)
        sql += " ORDER BY id LIMIT ?"
        params.append(size)
        with self._lock:
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return 0
            chunk_ids = {int(row["chunk_id"]) for row in rows}
            placeholders = ",".join("?" * len(chunk_ids))
            bodies = {
                int(row["id"]): str(row["text"] or "")
                for row in conn.execute(
                    f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
                    list(chunk_ids),
                ).fetchall()
            }
        texts = iter_window_texts(rows, bodies=bodies)
        name, dimensions = self._embedder_identity()
        try:
            vectors = self.embedder.embed_many(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("resource index drain could not embed a batch: %s", exc)
            return 0
        updates = [
            (pack_vector(vector), name, dimensions, int(row["id"]))
            for row, vector in zip(rows, vectors, strict=False)
        ]
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(
                    "UPDATE windows SET embedding=?, embedder=?, dimensions=? "
                    "WHERE id=? AND embedding IS NULL",
                    updates,
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return len(updates)

    # --- diagnostics --------------------------------------------------------

    def status(self, *, pot_id: str | None = None) -> ResourceIndexStatus:
        """Base status, with the vector arm's reason attached when it is down.

        Called unbound rather than through ``super()``: ``@dataclass(slots=True)``
        replaces the class object, and a zero-argument ``super()`` closes over
        the original — a trap worth spending one explicit name to avoid.
        """
        report = SqliteFtsResourceIndex.status(self, pot_id=pot_id)
        reason = self._vector_error()
        if reason is None or not report.ready:
            return report
        joined = "; ".join(part for part in (reason, report.detail) if part)
        return dataclasses.replace(report, detail=joined)


__all__ = [
    "EMBED_BATCH",
    "SqliteHybridResourceIndex",
    "load_vector_extension",
    "pack_vector",
]
