"""Engine queue selection without host imports or path mutation."""

from __future__ import annotations

import os

from potpie_context_engine.domain.ports.context_graph_job_queue import (
    ContextGraphJobQueuePort,
    NoOpContextGraphJobQueue,
)


def get_context_graph_job_queue(
    injected: ContextGraphJobQueuePort | None = None,
) -> ContextGraphJobQueuePort:
    """Return an injected queue or an explicit engine-owned adapter.

    The standalone default is inline/no-op. Embedding products inject their own
    Celery or broker adapter through ``EngineDependencies``; module paths and
    application imports are never resolved by the engine.
    """

    if injected is not None:
        return injected
    backend = (os.getenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND") or "noop").strip().lower()
    if backend in {"noop", "none", "disabled", "inline", "memory"}:
        return NoOpContextGraphJobQueue()
    if backend == "hatchet":
        from potpie_context_engine.adapters.outbound.hatchet.hatchet_job_queue import (
            HatchetContextGraphJobQueue,
        )

        return HatchetContextGraphJobQueue.from_env()
    if backend == "celery":
        raise ValueError(
            "Celery queues must be injected through EngineDependencies.job_queue"
        )
    raise ValueError(
        f"Unknown CONTEXT_GRAPH_JOB_QUEUE_BACKEND={backend!r}; expected noop or hatchet"
    )


__all__ = ["get_context_graph_job_queue"]
