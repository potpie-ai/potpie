"""Transport-only queue for ingestion pipeline (no business logic)."""

from __future__ import annotations

from typing import Protocol


class IngestionQueue(Protocol):
    """
    Enqueue durable work for event processing and step execution.

    Distinct from :class:`ContextGraphJobQueuePort` until migration converges adapters.
    """

    def enqueue_event(self, event_id: str) -> None:
        """Start or continue event processing (planning stage)."""
        ...

    def enqueue_step(self, event_id: str, step_id: str, pot_id: str) -> None:
        """Execute one planned step (execution stage)."""
        ...
