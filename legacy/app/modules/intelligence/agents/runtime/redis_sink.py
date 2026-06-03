"""RedisStreamManager-backed AgentRunSink.

Implements the AgentRunSink port over the synchronous RedisStreamManager. Used by
the Celery backend, and reused as the replay-cache writer for Hatchet runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.modules.conversations.utils.redis_streaming import RedisStreamManager


class RedisStreamSink:
    """AgentRunSink delegating to a RedisStreamManager for a fixed conversation/run."""

    def __init__(
        self, redis_manager: "RedisStreamManager", conversation_id: str, run_id: str
    ):
        self._redis = redis_manager
        self._conversation_id = conversation_id
        self._run_id = run_id

    def emit(self, event_type: str, payload: dict) -> None:
        self._redis.publish_event(
            self._conversation_id, self._run_id, event_type, payload
        )

    def set_status(self, status: str) -> None:
        self._redis.set_task_status(self._conversation_id, self._run_id, status)

    def is_cancelled(self) -> bool:
        return self._redis.check_cancellation(self._conversation_id, self._run_id)
