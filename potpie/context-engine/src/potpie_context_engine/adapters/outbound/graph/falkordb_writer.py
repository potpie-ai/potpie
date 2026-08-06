"""FalkorDB :class:`GraphWriterPort` adapter (lightweight local backend).

Same Position-B substrate as the Neo4j writer, same five write verbs. The
mutation Cypher (`MERGE` / `ON CREATE SET randomUUID()` / `timestamp()` /
dynamic `SET e:Label`) is identical on FalkorDB (verified by the Phase-0
spike), so this adapter **reuses** ``cypher.py``'s async mutation functions
through a thin async-driver shim over the synchronous ``falkordb`` client —
rather than duplicating the bitemporal / supersession logic.

Only three things are FalkorDB-specific and live here:

1. the async shim (``_FalkorAsyncDriver``) over ``falkordb``'s sync client;
2. ``ensure_indexes`` — FalkorDB rejects named indexes and ``IF NOT EXISTS``,
   so it uses the unnamed ``CREATE INDEX FOR ... ON (...)`` form, best-effort;
3. ``reset_pot`` — FalkorDB has no ``CALL {} IN TRANSACTIONS``, so it deletes
   in a client-side batched loop scoped to ``group_id``.

The default local backend is **FalkorDBLite** — an embedded Redis (via
``redislite``) backed by a local file, so ``pip install`` is enough (no server
or Docker). Server/container mode (``falkordb_url`` over a redis URL) is kept
as a deferred profile and requires the optional ``falkordb`` client.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import time
from typing import Any, Callable, Coroutine, TypeVar

from potpie_context_core.errors import GraphSubstrateUnavailable

from potpie_context_engine.adapters.outbound.graph.cypher import (
    _render_fact,
    _require_valid_pot_id,
    _stable_source_ref,
    apply_invalidations_async,
    delete_edges_async,
    upsert_edges_async,
    upsert_entities_async,
)
from potpie_context_engine.adapters.outbound.graph.writer_port import GraphWriterPort
from potpie_context_core.definition import DEFAULT_GRAPH_DEFINITION, GraphDefinition
from potpie_context_core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from potpie_context_engine.domain.retrieval_card import build_retrieval_card
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)

# Redis names the multi-part AOF's index file; its presence — not the
# directory's — is what marks a usable AOF (see ``_aof_is_ready``).
_AOF_DIRNAME = "appendonlydir"
_AOF_MANIFEST = "appendonly.aof.manifest"

# A rewrite of a local graph is sub-second; this bound only exists so a wedged
# rewrite degrades instead of hanging the CLI.
_AOF_REWRITE_TIMEOUT_S = 30.0
_AOF_POLL_S = 0.05

# ``redislite`` writes this handshake file beside the db so a second process
# attaches to the running server instead of spawning its own, and removes it
# when the owning process exits cleanly. Left behind, it is a record of a server
# that died without saving — see ``_died_without_saving``.
_SETTINGS_SUFFIX = ".settings"

# Embedded servers this process started, by db path. Kept because ``redislite``
# will not reliably stop them for us: its atexit hook shuts the server down only
# when it sees at most one connection, and any process that has issued
# concurrent queries — every daemon — has a larger pool by then and leaks the
# server instead. Owning the shutdown is the only way to make ``daemon stop``
# actually stop it.
_OWNED_SERVERS: dict[str, Any] = {}

_T = TypeVar("_T")

# Unnamed index DDL (FalkorDB rejects named indexes + IF NOT EXISTS). Mirrors
# the canonical indexes the Position-B traversal patterns rely on.
_INDEX_STATEMENTS = (
    "CREATE INDEX FOR (n:Entity) ON (n.group_id, n.entity_key)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.invalid_at)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name, r.valid_at)",
)

# reset_pot batch size for the client-side delete loop.
_RESET_BATCH = 500


def _index_statements(embedding_dim: int) -> tuple[str, ...]:
    return (
        *_INDEX_STATEMENTS,
        "CREATE VECTOR INDEX FOR ()-[r:RELATES_TO]->() ON (r.fact_embedding) "
        f"OPTIONS {{dimension:{int(embedding_dim)}, similarityFunction:'cosine'}}",
    )


def _records_from_result(result: Any) -> list[dict[str, Any]]:
    """Map a FalkorDB ``QueryResult`` to Neo4j-shaped record dicts.

    ``header`` is ``[[type_code, name], ...]`` and ``result_set`` is a list of
    rows; we key each row by its column alias so reused Neo4j code can do
    ``rec["props"]`` / ``rec["cnt"]`` unchanged.
    """
    header = getattr(result, "header", None) or []
    names = [h[1] if isinstance(h, (list, tuple)) and len(h) > 1 else h for h in header]
    rows = getattr(result, "result_set", None) or []
    return [dict(zip(names, row)) for row in rows]


class _FalkorAsyncResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    async def single(self) -> dict[str, Any] | None:
        return self._records[0] if self._records else None

    async def consume(self) -> None:
        return None


class _FalkorAsyncSession:
    """Async-shaped session over the sync ``falkordb`` graph handle.

    Mirrors the slice of the Neo4j async session API that ``cypher.py`` uses:
    ``async with``, ``await run(...)`` → result with ``await single()`` /
    ``await consume()``.
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def __aenter__(self) -> "_FalkorAsyncSession":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def run(self, cypher: str, **params: Any) -> _FalkorAsyncResult:
        def _run() -> list[dict[str, Any]]:
            result = self._graph.query(cypher, params=params or None)
            return _records_from_result(result)

        records = await asyncio.to_thread(_run)
        return _FalkorAsyncResult(records)


class _FalkorAsyncDriver:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def session(self) -> _FalkorAsyncSession:
        return _FalkorAsyncSession(self._graph)

    async def close(self) -> None:
        return None


def build_falkordb_graph(settings: ContextEngineSettingsPort) -> Any:
    """Build a FalkorDB graph handle from settings.

    Default (``lite``) → embedded FalkorDBLite via ``redislite``, backed by a
    local file: no server, no Docker. ``server`` mode → connect to a running
    FalkorDB over a redis URL (deferred profile; needs the optional
    ``falkordb`` client). Both expose the same ``graph.query(...)`` →
    ``result.header`` / ``result.result_set`` surface this adapter relies on.
    """
    name = settings.falkordb_graph_name()
    if settings.falkordb_mode() == "server":
        url = settings.falkordb_url()
        if not url:
            raise RuntimeError(
                "falkordb server mode requires a URL — set FALKORDB_URL "
                "(or CONTEXT_ENGINE_FALKORDB_URL), or use the default lite mode"
            )
        from falkordb import FalkorDB

        return FalkorDB.from_url(url).select_graph(name)
    # Lite (default): embedded FalkorDBLite over a local file — no server.
    from redislite.falkordb_client import FalkorDB as LiteFalkorDB

    path = settings.falkordb_lite_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _refuse_untrustworthy_state(path)
    db = LiteFalkorDB(path, serverconfig=_lite_server_config(path))
    _ensure_lite_durability(db, path)
    _remember_owned_server(db, path)
    return db.select_graph(name)


def _settings_file(path: str) -> str:
    return path + _SETTINGS_SUFFIX


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by somebody else.
        return True
    return True


def _died_without_saving(path: str) -> bool:
    """Did the last embedded server for this db die ungracefully?

    ``redislite`` removes its settings file as part of the shutdown its owning
    process runs at exit, so the file surviving with a dead pid behind it is an
    exact record of a server that was killed — OOM, ``kill -9``, a container
    stop, a laptop that slept and never came back. A missing file, or a live
    pid, means there is nothing to worry about.

    Deliberately conservative: an unreadable or malformed settings file counts
    as unclean, because the alternative is guessing in the direction that loses
    data quietly.
    """
    settings_file = _settings_file(path)
    if not os.path.isfile(settings_file):
        return False
    try:
        with open(settings_file) as handle:
            settings = json.load(handle)
        pidfile = settings["pidfile"]
        with open(pidfile) as handle:
            pid = int(handle.read().strip())
    except (OSError, ValueError, KeyError, TypeError):
        return True
    return not _pid_is_alive(pid)


def _refuse_untrustworthy_state(path: str) -> None:
    """Do not open a graph that may silently be missing its most recent writes.

    After an ungraceful death the next process attaches to nothing, spawns a
    fresh server, and loads whatever is on disk. With AOF that is everything.
    Without it — a db whose first-run migration never completed — the newest
    snapshot can predate committed writes, and the read that follows returns
    ``ok: true`` over a partial graph. That is strictly worse than an error: a
    memory product answering confidently from a dataset it cannot vouch for
    teaches the agent to trust an answer that is wrong.

    So: refuse, and name the recovery. Deleting the settings file is the
    acknowledgement — it is the record of the crash, and removing it says the
    operator accepts continuing from the last snapshot.
    """
    if not _died_without_saving(path) or _aof_is_ready(path):
        return
    raise GraphSubstrateUnavailable(
        f"the embedded graph server for {path} shut down without saving, and no "
        "crash-durable AOF was active at the time — the data on disk may be "
        "missing writes that were reported as committed, so this graph cannot "
        "be served as complete",
        recommended_next_action=(
            f"to continue from the last snapshot, accepting any writes lost in "
            f"the crash, delete {_settings_file(path)} and re-run"
        ),
    )


def _remember_owned_server(db: Any, path: str) -> None:
    """Record a server *this* process started, so it can be stopped later.

    Only servers we started: ``redislite`` sets ``cleanupregistry`` on the
    client that spawned the server, and a process that merely attached to a
    running one must never shut it down out from under its owner.

    The atexit hook covers every process, not just the daemon. Any process that
    runs two queries concurrently — the writer dispatches through
    ``asyncio.to_thread`` — ends up with a multi-connection pool, which is the
    exact condition under which ``redislite`` declines to stop the server it
    started. Without this, a CLI run against a scratch home leaves a server
    behind as reliably as the daemon does.
    """
    conn = getattr(db, "connection", None)
    if conn is None or not getattr(conn, "cleanupregistry", False):
        return
    if not _OWNED_SERVERS:
        atexit.register(shutdown_embedded_servers)
    _OWNED_SERVERS[path] = conn


def shutdown_embedded_servers() -> int:
    """Stop every embedded server this process started. Returns how many.

    Called on daemon shutdown. Without it, ``potpie daemon stop`` leaves a
    ``redis-server`` reparented to init, holding the db file and its memory,
    for as long as the machine is up — ``redislite``'s own atexit hook declines
    to stop a server with more than one connection open, which is every daemon.

    Best-effort and idempotent: a server that is already gone is a success.
    """
    stopped = 0
    for path, conn in list(_OWNED_SERVERS.items()):
        _OWNED_SERVERS.pop(path, None)
        try:
            conn.shutdown(save=True, now=True, force=True)
            stopped += 1
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.debug("falkordb_lite: server at %s did not stop (%s)", path, exc)
    return stopped


def _active_socket(path: str) -> str | None:
    """The unix socket of the server currently serving this db, if any."""
    try:
        with open(_settings_file(path)) as handle:
            return json.load(handle).get("unixsocket")
    except (OSError, ValueError, AttributeError):
        return None


def embedded_server_report(path: str) -> dict[str, Any]:
    """Which embedded graph servers are running, and which nobody owns.

    Every ``redislite`` server is daemonized, so it survives the process that
    started it — six were found running on one laptop, the oldest 15 days old,
    two of them against home directories that no longer existed. Each holds
    memory and a db file open forever.

    This *reports*; it does not kill. A server on another db path may belong to
    a different Potpie home, another checkout, or another user, and there is no
    way from here to tell a leak from somebody's live daemon. Reaping the ones
    it started is :func:`shutdown_embedded_servers`' job, at a point where
    ownership is known.
    """
    report: dict[str, Any] = {"running": 0, "for_this_graph": 0, "unattributed": []}
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil ships with redislite
        report["detail"] = "psutil unavailable; cannot enumerate embedded servers"
        return report

    ours = _active_socket(path)
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            name = proc.info.get("name") or ""
            if "redis-server" not in name:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "redislite" not in cmdline:
                continue
            report["running"] += 1
            if ours and ours in cmdline:
                report["for_this_graph"] += 1
                continue
            report["unattributed"].append(
                {
                    "pid": proc.info.get("pid"),
                    "started_at": proc.info.get("create_time"),
                }
            )
        except Exception:  # noqa: BLE001 - a process can exit mid-iteration
            continue
    if report["unattributed"]:
        report["detail"] = (
            f"{len(report['unattributed'])} embedded graph server(s) are running "
            "that this graph does not use; they may belong to another Potpie "
            "home, or be leftovers from processes that have exited"
        )
    return report


def _aof_dir(path: str) -> str:
    """Where Redis keeps the multi-part AOF — beside the db file, per its ``dir``."""
    return os.path.join(os.path.dirname(path) or ".", _AOF_DIRNAME)


def _aof_is_ready(path: str) -> bool:
    """True when a *complete* AOF exists for this db file.

    Deliberately tests the manifest rather than the directory. ``CONFIG SET
    appendonly yes`` creates the directory immediately and writes
    ``temp-appendonly.aof.incr`` while the rewrite runs in the background; a
    process that exits in that window leaves a directory holding only that temp
    file. Treating the directory as proof of an AOF would then make the next
    startup load an empty AOF *over* a perfectly good RDB — silently discarding
    the pot. The manifest only appears once the rewrite has completed, so an
    interrupted migration correctly falls back to the RDB and retries.
    """
    return os.path.isfile(os.path.join(_aof_dir(path), _AOF_MANIFEST))


def _lite_server_config(path: str) -> dict[str, str]:
    """Server config for the embedded instance, conditional on a complete AOF.

    Enabling ``appendonly`` at *startup* makes the AOF authoritative, and Redis
    treats a missing AOF as an empty dataset — so passing it on a db that only
    has an RDB silently discards every existing claim. The AOF is therefore
    only requested once one is known-complete; the first run migrates at
    runtime in :func:`_ensure_lite_durability`, rewriting from the loaded RDB.
    """
    if _aof_is_ready(path):
        return {"appendonly": "yes", "appendfsync": "always"}
    return {}


def _ensure_lite_durability(db: Any, path: str) -> None:
    """Turn on AOF for an embedded instance that does not have a complete one.

    Without this the lite profile persists **only** on a clean shutdown:
    ``redislite`` saves the RDB when the owning process exits, and the RDB rule
    (``save 900 1``) is the sole other trigger, so an ungraceful death — OOM,
    laptop sleep, SIGKILL — loses every write since the last save. Enabling AOF
    here rather than in the startup config is what makes the upgrade
    non-destructive: the RDB has already been loaded into memory, and
    ``CONFIG SET appendonly yes`` rewrites the AOF from that, so existing pots
    carry forward.

    The wait is load-bearing, not politeness. The rewrite is asynchronous, and
    a process that returns before it finishes can write claims and exit with
    the AOF still a temp file — the writes reach neither a finished AOF nor
    (on a crash) the RDB. Blocking until the manifest lands is what makes the
    first run after the upgrade as durable as every run after it.

    ``appendfsync always`` rather than ``everysec`` because Potpie writes a
    handful of claims per interaction — the per-write fsync is affordable at
    that volume, and it makes "committed" mean durable.

    Best-effort: a store that will not take the setting is still usable exactly
    as it was before, so this warns rather than failing startup.
    """
    if _aof_is_ready(path):
        return
    try:
        conn = db.connection
        conn.config_set("appendfsync", "always")
        conn.config_set("appendonly", "yes")
        deadline = time.monotonic() + _AOF_REWRITE_TIMEOUT_S
        while time.monotonic() < deadline:
            info = conn.info("persistence")
            if not info.get("aof_rewrite_in_progress") and _aof_is_ready(path):
                return
            time.sleep(_AOF_POLL_S)
        logger.warning(
            "falkordb_lite: AOF rewrite at %s did not complete within %.0fs; "
            "the graph persists only on clean shutdown until it does",
            _aof_dir(path),
            _AOF_REWRITE_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never block startup
        logger.warning(
            "falkordb_lite: could not enable AOF durability at %s (%s); "
            "the graph persists only on clean shutdown until this succeeds",
            _aof_dir(path),
            exc,
        )


class FalkorDBGraphProvider:
    """Lazily build and memoize **one** shared FalkorDB graph handle.

    The writer and reader must share a single handle so they talk to the same
    embedded FalkorDBLite instance (two handles on one db file would each spawn
    a redis-server). Lazy so ``build_container`` never connects at wiring time.
    """

    def __init__(self, settings: ContextEngineSettingsPort) -> None:
        self._settings = settings
        self._graph: Any | None = None

    def __call__(self) -> Any:
        if self._graph is None:
            self._graph = build_falkordb_graph(self._settings)
        return self._graph


class FalkorDBGraphWriter(GraphWriterPort):
    """Production-shaped writer for the lightweight FalkorDB local backend."""

    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        graph: Any | None = None,
        graph_provider: Callable[[], Any] | None = None,
        embedder: Any | None = None,
        definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
    ) -> None:
        self._settings = settings
        self._enabled = settings.is_enabled()
        self._graph = graph  # injectable for unit tests
        self._graph_provider = graph_provider  # shared handle from the container
        self._embedder = embedder
        self._definition = definition

    def bind_definition(self, definition: GraphDefinition) -> FalkorDBGraphWriter:
        return FalkorDBGraphWriter(
            self._settings,
            graph=self._graph,
            graph_provider=self._graph_provider,
            embedder=self._embedder,
            definition=definition,
        )

    @property
    def enabled(self) -> bool:
        if not self._enabled:
            return False
        # A directly-injected graph (unit tests) is always usable.
        if self._graph is not None:
            return True
        # Otherwise mirror exactly what build_falkordb_graph can honor, so the
        # gate never reports enabled for a config the builder would reject:
        # server mode needs a URL; lite (default) needs no external config.
        if self._settings.falkordb_mode() == "server":
            return bool(self._settings.falkordb_url())
        return self._settings.falkordb_mode() == "lite"

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = (
                self._graph_provider()
                if self._graph_provider is not None
                else build_falkordb_graph(self._settings)
            )
        return self._graph

    async def _with_driver(self, fn: Callable[[Any], Coroutine[Any, Any, _T]]) -> _T:
        driver = _FalkorAsyncDriver(self._get_graph())
        return await fn(driver)

    async def ensure_indexes(self) -> bool:
        if not self.enabled:
            return False
        graph = self._get_graph()
        embedding_dim = int(getattr(self._embedder, "dimensions", 1536))
        await asyncio.to_thread(self._ensure_indexes_sync, graph, embedding_dim)
        return True

    @staticmethod
    def _ensure_indexes_sync(graph: Any, embedding_dim: int = 1536) -> None:
        for stmt in _index_statements(embedding_dim):
            try:
                graph.query(stmt)
            except Exception as exc:  # noqa: BLE001
                # Re-running creates an "already indexed" error; best-effort.
                logger.debug("falkordb index skipped (%s): %s", stmt, exc)

    async def upsert_entities(
        self, pot_id: str, items: list[EntityUpsert], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        return await self._with_driver(
            lambda d: upsert_entities_async(
                d,
                pot_id,
                items,
                provenance,
                definition=self._definition,
            )
        )

    async def upsert_edges(
        self, pot_id: str, items: list[EdgeUpsert], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        written = await self._with_driver(
            lambda d: upsert_edges_async(
                d,
                pot_id,
                items,
                provenance,
                definition=self._definition,
            )
        )
        if self._embedder is not None and written:
            await asyncio.to_thread(
                self._write_edge_vectors_sync,
                self._get_graph(),
                pot_id,
                items,
                provenance,
            )
        return written

    async def delete_edges(
        self, pot_id: str, items: list[EdgeDelete], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        return await self._with_driver(
            lambda d: delete_edges_async(
                d,
                pot_id,
                items,
                provenance,
                definition=self._definition,
            )
        )

    async def invalidate(
        self, pot_id: str, items: list[InvalidationOp], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        return await self._with_driver(
            lambda d: apply_invalidations_async(
                d,
                pot_id,
                items,
                provenance,
                definition=self._definition,
            )
        )

    async def reset_pot(self, pot_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "context_graph_disabled"}
        try:
            _require_valid_pot_id(pot_id)
        except ValueError as exc:
            return {"ok": False, "error": f"invalid_pot_id: {exc}"}
        try:
            return await asyncio.to_thread(self._reset_pot_sync, pot_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reset_pot failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def _reset_pot_sync(self, pot_id: str) -> dict[str, Any]:
        graph = self._get_graph()
        before = self._count(graph, pot_id)
        # Client-side batched delete (no CALL {} IN TRANSACTIONS on FalkorDB).
        # The LIMIT guarantees forward progress; cap iterations defensively.
        # Count once up front, then re-count only after each delete batch.
        remaining = before
        for _ in range(before // _RESET_BATCH + 2):
            if remaining == 0:
                break
            graph.query(
                "MATCH (n {group_id: $gid}) WITH n LIMIT $lim DETACH DELETE n",
                params={"gid": pot_id, "lim": _RESET_BATCH},
            )
            remaining = self._count(graph, pot_id)
        if remaining:
            return {
                "ok": False,
                "error": "group_id_reset_incomplete",
                "group_id_nodes_before": before,
                "group_id_nodes_remaining": remaining,
            }
        return {
            "ok": True,
            "group_id_nodes_before": before,
            "group_id_nodes_remaining": 0,
        }

    @staticmethod
    def _count(graph: Any, pot_id: str) -> int:
        result = graph.query(
            "MATCH (n {group_id: $gid}) RETURN count(n) AS cnt",
            params={"gid": pot_id},
        )
        rows = getattr(result, "result_set", None) or []
        return int(rows[0][0]) if rows and rows[0] else 0

    def _write_edge_vectors_sync(
        self,
        graph: Any,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> None:
        if self._embedder is None:
            return
        for item in items:
            raw_props = dict(item.properties)
            source_ref = _stable_source_ref(
                predicate=item.edge_type,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                provenance=provenance,
            )
            if isinstance(raw_props.get("source_ref"), str) and raw_props["source_ref"]:
                source_ref = raw_props["source_ref"]
            fact = _render_fact(
                predicate=item.edge_type,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                extra=raw_props,
            )
            card = build_retrieval_card(
                description=raw_props.get("description")
                if isinstance(raw_props.get("description"), str)
                else None,
                fact=fact,
                subject_key=item.from_entity_key,
                predicate=item.edge_type,
                object_key=item.to_entity_key,
                scope=raw_props.get("code_scope")
                if isinstance(raw_props.get("code_scope"), dict)
                else None,
            )
            if not card:
                continue
            try:
                # Keep embedding inside the try: a model error must degrade to
                # "no vector enrichment", not abort the already-written edge.
                embedding = [float(x) for x in self._embedder.embed(card)]
                graph.query(
                    """
                    MATCH (:Entity {group_id: $gid, entity_key: $from_key})
                          -[r:RELATES_TO {
                              group_id: $gid,
                              name: $predicate,
                              subject_key: $from_key,
                              object_key: $to_key,
                              source_ref: $source_ref
                          }]->
                          (:Entity {group_id: $gid, entity_key: $to_key})
                    SET r.fact_embedding = vecf32($embedding),
                        r.embedding_model = $embedding_model,
                        r.embedding_dim = $embedding_dim
                    """,
                    params={
                        "gid": pot_id,
                        "predicate": item.edge_type,
                        "from_key": item.from_entity_key,
                        "to_key": item.to_entity_key,
                        "source_ref": source_ref,
                        "embedding": embedding,
                        "embedding_model": getattr(self._embedder, "name", "unknown"),
                        "embedding_dim": int(
                            getattr(self._embedder, "dimensions", len(embedding))
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "falkordb vector write skipped for %s:%s->%s: %s",
                    item.edge_type,
                    item.from_entity_key,
                    item.to_entity_key,
                    exc,
                )


__all__ = [
    "FalkorDBGraphProvider",
    "FalkorDBGraphWriter",
    "build_falkordb_graph",
    "embedded_server_report",
    "shutdown_embedded_servers",
]
