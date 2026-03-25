"""Optional async job enqueue (Celery, RQ, etc.) — port."""

from typing import Protocol


class JobEnqueuePort(Protocol):
    def enqueue_backfill(self, project_id: str) -> None:
        ...

    def enqueue_ingest_pr(self, project_id: str, pr_number: int) -> None:
        ...


class NoOpJobEnqueue:
    """CLI / in-process: no queue."""

    def enqueue_backfill(self, project_id: str) -> None:
        return None

    def enqueue_ingest_pr(self, project_id: str, pr_number: int) -> None:
        return None
