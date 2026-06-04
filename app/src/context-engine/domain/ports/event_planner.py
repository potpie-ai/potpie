"""Planning boundary: one canonical event -> ordered durable plan + steps (no graph writes)."""

from __future__ import annotations

from typing import Protocol

from domain.ingestion_event_models import IngestionEvent, PlanWithSteps


class EventPlanner(Protocol):
    """
    Load evidence via read-only connectors, produce ordered steps.

    Implementations must not write to Graphiti or the structural graph.
    """

    def plan(self, event: IngestionEvent) -> PlanWithSteps:
        ...
