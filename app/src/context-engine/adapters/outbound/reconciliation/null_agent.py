"""Placeholder agent that fails fast when no real planner is wired."""

from __future__ import annotations

from typing import Any

from domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from domain.reconciliation_batch import BatchAgentContext, BatchAgentOutcome


class NullReconciliationAgent:
    """Raises if invoked; install pydantic-deep + enable the planner flag for the real agent."""

    def run_batch(
        self,
        ctx: BatchAgentContext,
        *,
        checkpoints: AgentCheckpointStorePort | None = None,
        execution_log: object | None = None,
    ) -> BatchAgentOutcome:
        del ctx, checkpoints, execution_log
        raise NotImplementedError(
            "Reconciliation agent not configured; provide a host-backed ReconciliationAgentPort "
            "(install context-engine[reconciliation-agent])"
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {"agent": "null", "version": "0"}
