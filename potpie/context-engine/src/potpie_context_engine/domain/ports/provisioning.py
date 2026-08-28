"""Deployment-time provisioning contracts for graph backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from potpie_context_engine.core.ports.graph.backend import GraphBackend


@dataclass(frozen=True, slots=True)
class BackendProvisionResult:
    """Engine-owned result of preparing one graph backend's resources."""

    ok: bool
    detail: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProvisionableGraphBackend(GraphBackend, Protocol):
    """A graph backend that can provision its deployment resources."""

    def provision(self) -> BackendProvisionResult:
        """Stand up the backend's store idempotently."""
        ...


__all__ = ["BackendProvisionResult", "ProvisionableGraphBackend"]
