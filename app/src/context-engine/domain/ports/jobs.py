"""Backward-compatible re-exports for ``ContextGraphJobQueuePort``."""

from domain.ports.context_graph_job_queue import (
    ContextGraphJobQueuePort as JobEnqueuePort,
    NoOpContextGraphJobQueue as NoOpJobEnqueue,
)

__all__ = ["JobEnqueuePort", "NoOpJobEnqueue"]
