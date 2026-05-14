"""Potpie Celery adapter for ``ContextGraphJobQueuePort`` (``context-graph-etl`` queue)."""

from __future__ import annotations


class CeleryContextGraphJobQueue:
    """Enqueue Potpie Celery tasks registered in ``app.modules.context_graph.tasks``."""

    def enqueue_backfill(
        self, pot_id: str, *, target_repo_name: str | None = None
    ) -> None:
        from app.modules.context_graph.tasks import context_graph_backfill_pot

        context_graph_backfill_pot.delay(pot_id, target_repo_name=target_repo_name)

    def enqueue_batch(self, batch_id: str) -> None:
        from app.modules.context_graph.tasks import context_graph_process_batch

        context_graph_process_batch.delay(batch_id)
