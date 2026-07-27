"""Product-shell provisioning contract for graph backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from potpie_context_core.lifecycle import SetupPlan, StepResult
from potpie_context_core.ports.graph.backend import GraphBackend


@runtime_checkable
class ProvisionableGraphBackend(GraphBackend, Protocol):
    """A graph backend that can provision its deployment-time resources."""

    def provision(self, plan: SetupPlan) -> StepResult:
        """Stand up this backend's own store idempotently."""
        ...


__all__ = ["ProvisionableGraphBackend"]
