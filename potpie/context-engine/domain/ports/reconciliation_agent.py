"""Reconciliation agent execution (port)."""

from __future__ import annotations

from typing import Any, Protocol

from domain.reconciliation import ReconciliationPlan, ReconciliationRequest


class ReconciliationAgentPort(Protocol):
    def run_reconciliation(self, request: ReconciliationRequest) -> ReconciliationPlan:
        """Execute one reconciliation request; return a typed plan."""

    def capability_metadata(self) -> dict[str, Any]:
        """Optional descriptor for logging or host UI."""
        ...


# Phase A alias (INGESTION_ASYNC_PLAN): Ingestion Agent terminology
IngestionAgentPort = ReconciliationAgentPort
