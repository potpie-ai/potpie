"""Deployment-time provisioning contracts for graph backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from potpie_context_engine.core.lifecycle import SetupPlan, StepResult
from potpie_context_engine.core.ports.graph.backend import GraphBackend


@runtime_checkable
class ProvisionableGraphBackend(GraphBackend, Protocol):
    """A graph backend that can provision its deployment resources."""

    def provision(self, plan: SetupPlan) -> StepResult:
        """Stand up the backend's store idempotently."""
        ...


__all__ = ["ProvisionableGraphBackend"]
