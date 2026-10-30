"""Neo4j graph writer adapter.

One writer, one substrate: the reconciliation agent emits a typed
:class:`MutationBatch` and ``apply_mutation_batch`` calls these
methods directly. No episodic narrative tier, no LLM extraction, no
sync→async bridge — the agent decides what changed, the writer executes
the deterministic plan.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Coroutine, Iterable, TypeVar
from uuid import uuid4

from potpie_context_engine.adapters.outbound.graph.cypher import (
    _require_valid_pot_id,
    apply_invalidations_async,
    delete_edges_async,
    ensure_canonical_indexes,
    upsert_edges_async,
    upsert_entities_async,
)
from potpie_context_engine.adapters.outbound.graph.writer_port import GraphWriterPort
from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.claim_codec import row_from_record
from potpie_context_engine.core.ports.claim_query import ClaimRow
from potpie_context_engine.core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
    ProvenanceRef,
)
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class Neo4jGraphWriter(GraphWriterPort):
    """Production writer: applies typed mutations as Position-B :RELATES_TO edges."""

    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        embedder: Any | None = None,
        definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
        transaction: Any | None = None,
    ) -> None:
        self._settings = settings
        self._enabled = settings.is_enabled()
        self._embedder = embedder
        self._definition = definition
        self._transaction = transaction

    def bind_definition(self, definition: GraphDefinition) -> Neo4jGraphWriter:
        return Neo4jGraphWriter(
            self._settings,
            embedder=self._embedder,
            definition=definition,
            transaction=self._transaction,
        )

    def in_transaction(self, transaction: Any) -> Neo4jGraphWriter:
        return Neo4jGraphWriter(
            self._settings,
            embedder=self._embedder,
            definition=self._definition,
            transaction=transaction,
        )

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
        if self._transaction is not None:
            return await fn(_TransactionDriver(self._transaction))
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
        if not provenance.mutation_id:
            provenance = replace(provenance, mutation_id=uuid4().hex)
        from .neo4j_inventory import prepare_entities, validate_aliases

        async def write(transaction):
            resolved = await prepare_entities(transaction, pot_id, items, provenance)
            count = await upsert_entities_async(
                _TransactionDriver(transaction),
                pot_id,
                resolved,
                provenance,
                definition=self._definition,
            )
            await validate_aliases(transaction, pot_id)
            return count

        if self._transaction is not None:
            return await write(self._transaction)

        async def own_transaction(driver):
            async with driver.session() as session:
                return await session.execute_write(write)

        return await self._with_driver(own_transaction)

    async def claim_rows(
        self,
        pot_id: str,
        *,
        record_ids: Iterable[str] = (),
        history_scopes: Iterable[tuple[str, str]] = (),
        corrections: Iterable[EdgeDelete | InvalidationOp] = (),
    ) -> list[ClaimRow]:
        """Read selected histories through the mutation's borrowed transaction."""
        from .claim_reads import claim_read_query

        _require_valid_pot_id(pot_id)
        query = claim_read_query(
            pot_id,
            record_ids=record_ids,
            history_scopes=history_scopes,
            corrections=corrections,
        )
        if query is None:
            return []
        cypher, params = query

        async def read(driver: Any) -> list[ClaimRow]:
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                return [row_from_record(record) async for record in result]

        return await self._with_driver(read)

    async def upsert_edges(
        self, pot_id: str, items: list[EdgeUpsert], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        return await self._with_driver(
            lambda d: upsert_edges_async(
                d,
                pot_id,
                items,
                provenance,
                embedder=self._embedder,
                definition=self._definition,
            )
        )

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

    async def reset_pot(
        self, pot_id: str, *, mutation_id: str | None = None
    ) -> dict[str, Any]:
        reset_id = mutation_id if mutation_id is not None else uuid4().hex
        pending = {"mutation_id": reset_id, "execution": None}
        if not self.enabled:
            return {"ok": False, "error": "context_graph_disabled", **pending}
        try:
            _require_valid_pot_id(pot_id)
        except ValueError as exc:
            return {"ok": False, "error": f"invalid_pot_id: {exc}", **pending}

        try:
            if not isinstance(reset_id, str) or not reset_id:
                raise ValueError("reset mutation_id must be a nonempty string")
            if self._transaction is not None:
                raise RuntimeError(
                    "reset must use the owning canonical mutation transaction"
                )
            from .neo4j_atomic import apply_atomic
            from potpie_context_engine.core.reconciliation import MutationBatch

            result = await apply_atomic(
                self,
                MutationBatch(reset_pot=True),
                expected_pot_id=pot_id,
                provenance_context=ProvenanceContext(mutation_id=reset_id),
                definition=self._definition,
            )
            return {
                "ok": result.ok,
                **result.administrative_summary,
                "mutation_id": result.mutation_id,
                "execution": result.execution.to_dict() if result.execution else None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("reset_pot failed: %s", exc)
            return {"ok": False, "error": str(exc), **pending}


class _TransactionSession:
    """Let existing Cypher helpers share a caller-owned transaction."""

    def __init__(self, transaction: Any) -> None:
        self.transaction = transaction

    async def __aenter__(self):
        return self.transaction

    async def __aexit__(self, *exc):
        return False


class _TransactionDriver:
    def __init__(self, transaction: Any) -> None:
        self.transaction = transaction

    def session(self):
        return _TransactionSession(self.transaction)


__all__ = ["GraphWriterPort", "Neo4jGraphWriter"]
