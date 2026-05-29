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

FalkorDBLite (embedded) is deferred; ``enabled`` requires a configured
``falkordb_url`` (server/container mode).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

from adapters.outbound.graph.cypher import (
    _require_valid_pot_id,
    apply_invalidations_async,
    delete_edges_async,
    upsert_edges_async,
    upsert_entities_async,
)
from adapters.outbound.graph.neo4j_writer import GraphWriterPort
from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Unnamed index DDL (FalkorDB rejects named indexes + IF NOT EXISTS). Mirrors
# the canonical indexes the Position-B traversal patterns rely on; the vector
# index is intentionally omitted (Neo4j syntax is rejected and reads use the
# Python token-overlap fallback).
_INDEX_STATEMENTS = (
    "CREATE INDEX FOR (n:Entity) ON (n.group_id, n.entity_key)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.invalid_at)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name, r.valid_at)",
)

# reset_pot batch size for the client-side delete loop.
_RESET_BATCH = 500


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


class FalkorDBGraphWriter(GraphWriterPort):
    """Production-shaped writer for the lightweight FalkorDB local backend."""

    def __init__(
        self, settings: ContextEngineSettingsPort, *, graph: Any | None = None
    ) -> None:
        self._settings = settings
        self._enabled = settings.is_enabled()
        self._graph = graph  # injectable for unit tests

    @property
    def enabled(self) -> bool:
        if not self._enabled:
            return False
        # Server/container mode only for now (Lite deferred): a URL must be set.
        return bool(self._graph is not None or self._settings.falkordb_url())

    def _get_graph(self) -> Any:
        if self._graph is None:
            url = self._settings.falkordb_url()
            if not url:
                raise RuntimeError("falkordb_unavailable")
            from falkordb import FalkorDB

            self._graph = FalkorDB.from_url(url).select_graph(
                self._settings.falkordb_graph_name()
            )
        return self._graph

    async def _with_driver(
        self, fn: Callable[[Any], Coroutine[Any, Any, _T]]
    ) -> _T:
        driver = _FalkorAsyncDriver(self._get_graph())
        return await fn(driver)

    async def ensure_indexes(self) -> bool:
        if not self.enabled:
            return False
        graph = self._get_graph()
        await asyncio.to_thread(self._ensure_indexes_sync, graph)
        return True

    @staticmethod
    def _ensure_indexes_sync(graph: Any) -> None:
        for stmt in _INDEX_STATEMENTS:
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
        return await self._with_driver(
            lambda d: upsert_edges_async(d, pot_id, items, provenance)
        )

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
        max_iters = before // _RESET_BATCH + 2
        for _ in range(max_iters):
            if self._count(graph, pot_id) == 0:
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


__all__ = ["FalkorDBGraphWriter"]
