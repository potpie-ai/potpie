"""Reconciliation agent execution (port)."""

from __future__ import annotations

from typing import Any, Protocol

from domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from domain.reconciliation_batch import BatchAgentContext, BatchAgentOutcome


class ReconciliationAgentPort(Protocol):
    def run_batch(
        self,
        ctx: BatchAgentContext,
        *,
        checkpoints: AgentCheckpointStorePort | None = None,
    ) -> BatchAgentOutcome:
        """Process a debounced batch of events to completion.

        The agent owns the loop: it calls graph-mutation, query, and integration
        tools as needed and persists progress via the optional ``checkpoints``
        port (after every tool call) so a worker crash mid-run can be recovered
        on the next dispatch via ``ctx.prior_messages_json``.
        """

    def capability_metadata(self) -> dict[str, Any]:
        """Optional descriptor for logging or host UI."""
        ...
