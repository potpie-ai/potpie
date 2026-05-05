"""Potpie Celery adapter for ``ContextGraphJobQueuePort`` (``context-graph-etl`` queue)."""

from __future__ import annotations


class CeleryContextGraphJobQueue:
    """Enqueue Potpie Celery tasks registered in ``app.modules.context_graph.tasks``."""

    def enqueue_backfill(
        self, pot_id: str, *, target_repo_name: str | None = None
    ) -> None:
        from app.modules.context_graph.tasks import context_graph_backfill_pot

        context_graph_backfill_pot.delay(pot_id, target_repo_name=target_repo_name)

    def enqueue_ingest_pr(
        self,
        pot_id: str,
        pr_number: int,
        *,
        is_live_bridge: bool = True,
        repo_name: str | None = None,
    ) -> None:
        from app.modules.context_graph.tasks import context_graph_ingest_pr

        context_graph_ingest_pr.delay(
            pot_id,
            pr_number,
            is_live_bridge=is_live_bridge,
            repo_name=repo_name,
        )

    def enqueue_ingestion_event(self, event_id: str, *, pot_id: str, kind: str) -> None:
        del pot_id, kind
        from app.modules.context_graph.tasks import context_graph_ingestion_agent_run

        context_graph_ingestion_agent_run.delay(event_id)

    def enqueue_episode_apply(self, pot_id: str, event_id: str, sequence: int) -> None:
        from app.modules.context_graph.tasks import context_graph_apply_episode

        context_graph_apply_episode.delay(pot_id, event_id, sequence)
