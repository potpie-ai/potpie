"""Graph mutation-plan persistence port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from potpie_context_core.domain.graph_plans import GraphMutationPlanRecord


class GraphPlanStorePort(Protocol):
    """Persist server-created mutation plans until commit, expiry, or close."""

    def save(self, record: GraphMutationPlanRecord) -> None:
        """Insert or replace a plan record."""
        ...

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        """Return one plan for a pot, if present."""
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


__all__ = ["GraphPlanStorePort"]
