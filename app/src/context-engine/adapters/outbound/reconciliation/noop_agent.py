"""No-op reconciliation agent for phase-2 smoke testing.

Satisfies :class:`ReconciliationAgentPort` but does no LLM call and no graph
mutation — it simply marks every event in the batch as completed. Used to
verify that the ingest → debounce → dispatch → process plumbing works
end-to-end before the real (pydantic-deep) agent is wired in phase 3.
"""

from __future__ import annotations

from typing import Any

from domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from domain.reconciliation_batch import BatchAgentContext, BatchAgentOutcome


class NoOpReconciliationAgent:
    """Marks all events in the batch as processed; no graph writes, no LLM."""

    def run_batch(
        self,
        ctx: BatchAgentContext,
        *,
        checkpoints: AgentCheckpointStorePort | None = None,
    ) -> BatchAgentOutcome:
        del checkpoints
        return BatchAgentOutcome(
            ok=True,
            completed_event_ids=[ev.event_id for ev in ctx.events],
            tool_call_count=0,
            error=None,
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "agent": "noop",
            "version": "phase2",
            "toolset_version": "none",
        }
