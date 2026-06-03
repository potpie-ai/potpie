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
        """Enqueue full pot backfill (historical PR ingestion, etc.)."""

    def enqueue_ingest_pr(
        self,
        pot_id: str,
        pr_number: int,
        *,
        is_live_bridge: bool = True,
        repo_name: str | None = None,
    ) -> None:
        """Enqueue ingest for a single pull request."""

    def enqueue_ingestion_event(self, event_id: str, *, pot_id: str, kind: str) -> None:
        """Run ingestion agent (or raw routing) for a persisted ``context_events`` row."""

    def enqueue_episode_apply(self, pot_id: str, event_id: str, sequence: int) -> None:
        """Apply one durable episode step (ordered per ``event_id`` / ``pot_id``)."""


class NoOpContextGraphJobQueue:
    """CLI / tests: accept enqueue calls but perform no broker I/O."""

    def enqueue_backfill(
        self, pot_id: str, *, target_repo_name: str | None = None
    ) -> None:
        return None

    def enqueue_ingest_pr(
        self,
        pot_id: str,
        pr_number: int,
        *,
        is_live_bridge: bool = True,
        repo_name: str | None = None,
    ) -> None:
        return None

    def enqueue_ingestion_event(self, event_id: str, *, pot_id: str, kind: str) -> None:
        return None

    def enqueue_episode_apply(self, pot_id: str, event_id: str, sequence: int) -> None:
        return None
