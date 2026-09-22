"""Graph mutation-plan persistence port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from potpie_context_core.graph_plans import GraphMutationPlanRecord


@runtime_checkable
class GraphPlanStorePort(Protocol):
    """Persist server-created mutation plans until commit, expiry, or close."""

    def save(self, record: GraphMutationPlanRecord) -> None:
        """Insert or replace a plan record."""
        ...

    def reserve_idempotency(
        self,
        *,
        record: GraphMutationPlanRecord,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> tuple[GraphMutationPlanRecord, bool]:
        """Atomically reserve a request key; return record and whether inserted."""
        ...

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        """Return one plan for a pot, if present."""
        ...

    def compare_and_set(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        """Atomically replace ``expected`` if it is still the stored record."""
        ...

    def list(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]:
        """Return plan records for history and operator inspection."""
        ...


@runtime_checkable
class AsyncGraphPlanStorePort(Protocol):
    async def save_async(self, record: GraphMutationPlanRecord) -> None: ...

    async def reserve_idempotency_async(
        self,
        *,
        record: GraphMutationPlanRecord,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> tuple[GraphMutationPlanRecord, bool]: ...

    async def get_async(
        self, *, pot_id: str, plan_id: str
    ) -> GraphMutationPlanRecord | None: ...

    async def compare_and_set_async(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool: ...

    async def list_async(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]: ...


__all__ = ["AsyncGraphPlanStorePort", "GraphPlanStorePort"]
