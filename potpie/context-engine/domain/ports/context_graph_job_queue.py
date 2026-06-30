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

    def enqueue_batch(self, batch_id: str) -> None:
        """Enqueue processing of a single reconciliation batch by id.

        Fired from ``admit_event`` on every successful event admission.
        Redundant calls for an already-claimed batch are a no-op on the
        worker side (``claim_batch_by_id`` returns ``None``).

        This is the sole enqueue path. Backfill is not a separate job: a
        source attach emits a single ``agent_reconciliation`` event that
        flows through ``admit_event`` → a batch → here, like any other
        event. The reconciliation agent (planner on, backfill playbooks)
        does the enumerate-and-seed work inside that batch.
        """


class NoOpContextGraphJobQueue:
    """CLI / tests: accept enqueue calls but perform no broker I/O."""

    def enqueue_batch(self, batch_id: str) -> None:
        return None
