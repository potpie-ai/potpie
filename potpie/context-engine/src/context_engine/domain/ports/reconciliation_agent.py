"""Reconciliation agent execution (port)."""

from __future__ import annotations

from typing import Any, Protocol

from context_engine.domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from context_engine.domain.ports.agent_execution_log import AgentExecutionLogPort
from context_engine.domain.reconciliation_batch import BatchAgentContext, BatchAgentOutcome


class ReconciliationAgentPort(Protocol):
    def run_batch(
        self,
        ctx: BatchAgentContext,
        *,
        checkpoints: AgentCheckpointStorePort | None = None,
        execution_log: AgentExecutionLogPort | None = None,
    ) -> BatchAgentOutcome:
        """Process a debounced batch of events to completion.

        The agent owns the loop: it calls graph-mutation, query, and
        integration tools as needed and streams its work + checkpoints
        progress into the durable ``execution_log`` (after every tool call)
        so a worker crash mid-run resumes with full context + the durable
        completed-event set via ``ctx.prior_messages_json`` /
        ``ctx.resume_completed_event_ids``. ``checkpoints`` is retained for
        call-site compatibility; the execution log subsumes it.
        """
        ...

    def capability_metadata(self) -> dict[str, Any]:
        """Optional descriptor for logging or host UI."""
        ...
