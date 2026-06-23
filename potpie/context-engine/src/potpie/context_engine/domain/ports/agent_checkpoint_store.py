"""Agent checkpoint store (port).

Stores the agent's serialized message history per batch so that, if a worker
crashes mid-run, the next worker that picks up the batch can resume the agent
from the last completed tool call instead of restarting.
"""

from __future__ import annotations

from typing import Any, Protocol

from potpie.context_engine.domain.reconciliation_batch import AgentCheckpoint


class AgentCheckpointStorePort(Protocol):
    def load(self, batch_id: str) -> AgentCheckpoint | None: ...

    def save(
        self,
        batch_id: str,
        *,
        messages_json: list[dict[str, Any]],
        tool_call_count: int,
    ) -> None:
        """Upsert the checkpoint for ``batch_id``. Called after each tool call so
        a crash leaves at most one tool-call's worth of progress un-replayed."""

    def delete(self, batch_id: str) -> None:
        """Clear the checkpoint after the batch terminates successfully."""
