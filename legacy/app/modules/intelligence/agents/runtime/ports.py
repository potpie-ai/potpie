"""Backend-agnostic ports for agent execution.

The Celery and Hatchet backends each implement these so the shared run-loop and
routing logic stay decoupled from the queue/transport.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentRunSink(Protocol):
    """Sink for a single agent run's stream events, status, and cancellation signal.

    Celery implements this over ``RedisStreamManager``; Hatchet over
    ``ctx.aio_put_stream`` plus the Redis replay cache.
    """

    def emit(self, event_type: str, payload: dict) -> None:
        """Publish a stream event (``start`` | ``chunk`` | ``end``)."""
        ...

    def set_status(self, status: str) -> None:
        """Record run status (``queued`` | ``running`` | ``completed`` | ``cancelled`` | ``error``)."""
        ...

    def is_cancelled(self) -> bool:
        """True if the run has been asked to stop."""
        ...
