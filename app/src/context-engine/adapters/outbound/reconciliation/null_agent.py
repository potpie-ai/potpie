"""Placeholder agent when no LLM-backed planner is wired."""

from __future__ import annotations

from typing import Any

from domain.reconciliation import ReconciliationPlan, ReconciliationRequest


class NullReconciliationAgent:
    """Raises if invoked; use only when host overrides with a real ``ReconciliationAgentPort``."""

    def run_reconciliation(self, request: ReconciliationRequest) -> ReconciliationPlan:
        raise NotImplementedError(
            "Reconciliation agent not configured; provide a host-backed ReconciliationAgentPort"
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {"agent": "null", "version": "0"}
