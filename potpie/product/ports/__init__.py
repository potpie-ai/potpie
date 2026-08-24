"""The three service ports the host hosts.

The same three service modules run inside either the local daemon or the
managed API server:

    GraphService            data plane   — resolve / search / record
    PotManagementService    control plane — pots, active pot, sources, readiness
    SkillManager            CLI-managed agent skill catalog + nudge

``AgentContextPort`` (``potpie_context_engine.core.ports.agent_context``) composes them into the
public four-tool surface.
"""

from __future__ import annotations

from potpie.product.ports.auth import AuthIdentity, AuthService
from potpie_context_engine.core.ports.graph_service import (
    DataPlaneStatus,
    GraphService,
)
from potpie.product.ports.setup import SetupOrchestrator

__all__ = [
    "AuthIdentity",
    "AuthService",
    "DataPlaneStatus",
    "GraphService",
    "SetupOrchestrator",
]
