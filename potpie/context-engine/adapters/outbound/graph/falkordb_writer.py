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
import logging
import os
from typing import Any, Callable, Coroutine, TypeVar

from adapters.outbound.graph.cypher import (
    _render_fact,
    _require_valid_pot_id,
    _stable_source_ref,
    apply_invalidations_async,
    delete_edges_async,
    upsert_edges_async,
    upsert_entities_async,
)
from adapters.outbound.graph.writer_port import GraphWriterPort
from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from domain.retrieval_card import build_retrieval_card
from domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)

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
    return LiteFalkorDB(path).select_graph(name)


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
    ) -> None:
        self._settings = settings
        self._enabled = settings.is_enabled()
        self._graph = graph  # injectable for unit tests
        self._graph_provider = graph_provider  # shared handle from the container
        self._embedder = embedder
        self._indexes_ensured = False

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
            lambda d: upsert_entities_async(d, pot_id, items, provenance)
        )

    async def upsert_edges(
        self, pot_id: str, items: list[EdgeUpsert], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        written = await self._with_driver(
            lambda d: upsert_edges_async(d, pot_id, items, provenance)
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
            lambda d: delete_edges_async(d, pot_id, items, provenance)
        )

    async def invalidate(
        self, pot_id: str, items: list[InvalidationOp], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        return await self._with_driver(
            lambda d: apply_invalidations_async(d, pot_id, items, provenance)
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
    ) -> list[dict[str, str]]:
        """Attach fact embeddings to just-upserted edges.

        Returns one failure record per edge whose embedding could not be
        attached — a zero-row MATCH counts as a failure, not a no-op: the edge
        exists (the upsert wrote it), so silence here is how claims used to
        vanish from vector-only reads.
        """
        if self._embedder is None:
            return []
        # Setup's provision step normally creates the vector index, but a
        # write must not depend on setup having run (fresh homes, tests,
        # library embedding): ensure indexes once per writer so the vectors
        # written below are actually queryable.
        if not self._indexes_ensured:
            embedding_dim = int(getattr(self._embedder, "dimensions", 1536))
            self._ensure_indexes_sync(graph, embedding_dim)
            self._indexes_ensured = True
        failures: list[dict[str, str]] = []
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
                # The edge is matched with UNBOUND endpoints: anchoring the
                # source node (`(:Entity {entity_key: $from_key})-[r]->`) is
                # the embedded-FalkorDB plan shape that silently matches zero
                # rows when the source is internal node id 0 — always the
                # Repository on a fresh pot. See FIND_CLAIMS_CYPHER.
                result = graph.query(
                    """
                    MATCH ()-[r:RELATES_TO {
                              group_id: $gid,
                              name: $predicate,
                              subject_key: $from_key,
                              object_key: $to_key,
                              source_ref: $source_ref
                          }]->()
                    SET r.fact_embedding = vecf32($embedding),
                        r.embedding_model = $embedding_model,
                        r.embedding_dim = $embedding_dim
                    RETURN count(r) AS updated
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
                records = _records_from_result(result)
                updated = int(records[0]["updated"]) if records else 0
                if updated < 1:
                    raise RuntimeError(
                        "embedding attach matched no edge (upsert wrote it, "
                        "the attach MATCH could not find it)"
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "predicate": item.edge_type,
                        "subject_key": item.from_entity_key,
                        "object_key": item.to_entity_key,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "falkordb embedding attach FAILED for %s:%s->%s "
                    "(claim stays readable, semantic ranking degraded): %s",
                    item.edge_type,
                    item.from_entity_key,
                    item.to_entity_key,
                    exc,
                )
        if failures:
            logger.warning(
                "falkordb embedding attach failed for %d of %d edge(s) in this "
                "batch — run 'potpie graph repair semantic_index' to re-embed",
                len(failures),
                len(items),
            )
        return failures


__all__ = ["FalkorDBGraphProvider", "FalkorDBGraphWriter", "build_falkordb_graph"]
