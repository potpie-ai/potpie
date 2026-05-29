"""Queue port for context-graph background jobs (Celery, Hatchet, etc.)."""

from __future__ import annotations

from typing import Protocol


class ContextGraphJobQueuePort(Protocol):
    """
    Outbound port: enqueue context-graph work without binding to a specific broker.

    Implementations may use Celery (default), Hatchet (optional; self-hosted per
    https://docs.hatchet.run/self-hosting), or another backend; hosts select via
    ``bootstrap.queue_factory.get_context_graph_job_queue`` and ``CONTEXT_GRAPH_JOB_QUEUE_BACKEND``.
    """

    def enqueue_backfill(
        self, pot_id: str, *, target_repo_name: str | None = None
    ) -> None:
        """Enqueue full pot backfill (connector enumerate-then-submit sweep)."""

    def enqueue_batch(self, batch_id: str) -> None:
        """Enqueue processing of a single reconciliation batch by id.

        Fired from ``admit_event`` on every successful event admission.
        Redundant calls for an already-claimed batch are a no-op on the
        worker side (``claim_batch_by_id`` returns ``None``).
        """


class NoOpContextGraphJobQueue:
    """CLI / tests: accept enqueue calls but perform no broker I/O."""

    def enqueue_backfill(
        self, pot_id: str, *, target_repo_name: str | None = None
    ) -> None:
        return None

    def enqueue_batch(self, batch_id: str) -> None:
        return None
