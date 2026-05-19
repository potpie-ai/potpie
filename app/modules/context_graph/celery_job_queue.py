"""Potpie Celery adapter for ``ContextGraphJobQueuePort`` (``context-graph-etl`` queue)."""

from __future__ import annotations


class CeleryContextGraphJobQueue:
    """Enqueue Potpie Celery tasks registered in ``app.modules.context_graph.tasks``."""

    def enqueue_batch(self, batch_id: str) -> None:
        from app.modules.context_graph.tasks import context_graph_process_batch

        context_graph_process_batch.delay(batch_id)
