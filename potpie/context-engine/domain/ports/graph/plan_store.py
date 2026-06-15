"""Graph mutation-plan persistence port."""

from __future__ import annotations

from typing import Protocol

from domain.graph_plans import GraphMutationPlanRecord


class GraphPlanStorePort(Protocol):
    """Persist server-created mutation plans until commit, expiry, or close."""

    def save(self, record: GraphMutationPlanRecord) -> None:
        """Insert or replace a plan record."""
        ...

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        """Return one plan for a pot, if present."""
        ...


__all__ = ["GraphPlanStorePort"]
