"""No-op reconciliation agent — a test double, not a deployment option.

Satisfies :class:`ReconciliationAgentPort` but does no LLM call and no graph
mutation — it simply marks every event in the batch as completed. Tests use
it to verify the ingest → debounce → dispatch → process plumbing without a
real (pydantic-deep) agent; production wiring never constructs it.
"""

from __future__ import annotations

from typing import Any

from potpie_context_engine.domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from potpie_context_engine.domain.reconciliation_batch import BatchAgentContext, BatchAgentOutcome


class NoOpReconciliationAgent:
    """Marks all events in the batch as processed; no graph writes, no LLM."""

    def run_batch(
        self,
        ctx: BatchAgentContext,
        *,
        checkpoints: AgentCheckpointStorePort | None = None,
        execution_log: object | None = None,
    ) -> BatchAgentOutcome:
        del checkpoints, execution_log
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
