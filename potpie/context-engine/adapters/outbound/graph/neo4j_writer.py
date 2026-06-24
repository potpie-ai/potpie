"""Neo4j graph writer adapter.

One writer, one substrate: the reconciliation agent emits a typed
:class:`ReconciliationPlan` and ``apply_reconciliation_plan`` calls these
methods directly. No episodic narrative tier, no LLM extraction, no
sync→async bridge — the agent decides what changed, the writer executes
the deterministic plan.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, TypeVar

from adapters.outbound.graph.cypher import (
    _require_valid_pot_id,
    apply_invalidations_async,
    delete_edges_async,
    ensure_canonical_indexes,
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
from domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class Neo4jGraphWriter(GraphWriterPort):
    """Production writer: applies typed mutations as Position-B :RELATES_TO edges."""

    def __init__(
        self, settings: ContextEngineSettingsPort, *, embedder: Any | None = None
    ) -> None:
        self._settings = settings
        self._enabled = settings.is_enabled()
        self._embedder = embedder

    @property
    def enabled(self) -> bool:
        if not self._enabled:
            return False
        return bool(
            self._settings.neo4j_uri()
            and self._settings.neo4j_user() is not None
            and self._settings.neo4j_password() is not None
        )

    def _new_driver(self):
        from neo4j import AsyncGraphDatabase

        uri = self._settings.neo4j_uri()
        user = self._settings.neo4j_user()
        password = self._settings.neo4j_password()
        if not uri or user is None or password is None:
            return None
        return AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def _with_driver(self, fn: Callable[[Any], Coroutine[Any, Any, _T]]) -> _T:
        driver = self._new_driver()
        if driver is None:
            raise RuntimeError("neo4j_unavailable")
        try:
            return await fn(driver)
        finally:
            await driver.close()

    async def ensure_indexes(self) -> bool:
        if not self.enabled:
            return False
        embedding_dim = int(getattr(self._embedder, "dimensions", 1536))
        await self._with_driver(
            lambda d: ensure_canonical_indexes(d, embedding_dim=embedding_dim)
        )
        return True

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
            lambda d: upsert_edges_async(
                d, pot_id, items, provenance, embedder=self._embedder
            )
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

        async def _do(driver: Any) -> dict[str, Any]:
            async with driver.session() as session:
                res = await session.run(
                    "MATCH (n {group_id: $gid}) RETURN count(n) AS cnt", gid=pot_id
                )
                rec = await res.single()
                await res.consume()
                before = int(rec["cnt"]) if rec is not None else 0

                sweep = await session.run(
                    """
                    MATCH (n {group_id: $gid})
                    CALL (n) {
                        DETACH DELETE n
                    } IN TRANSACTIONS OF $batch ROWS
                    """,
                    gid=pot_id,
                    batch=500,
                )
                await sweep.consume()

                vres = await session.run(
                    "MATCH (n {group_id: $gid}) RETURN count(n) AS cnt", gid=pot_id
                )
                vrec = await vres.single()
                await vres.consume()
                remaining = int(vrec["cnt"]) if vrec is not None else 0
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

        try:
            return await self._with_driver(_do)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reset_pot failed: %s", exc)
            return {"ok": False, "error": str(exc)}


__all__ = ["GraphWriterPort", "Neo4jGraphWriter"]
