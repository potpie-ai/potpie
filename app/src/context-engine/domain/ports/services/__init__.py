"""The three service ports the host hosts.

The same three service modules run inside either the local daemon or the
managed API server:

    GraphService            data plane   — resolve / search / record
    PotManagementService    control plane — pots, active pot, sources, readiness
    SkillManager            CLI-managed agent skill catalog + nudge

``AgentContextPort`` (``domain.ports.agent_context``) composes them into the
public four-tool surface.
"""

from __future__ import annotations

from domain.ports.services.auth import AuthIdentity, AuthService
from domain.ports.services.config import ConfigService
from domain.ports.services.graph_service import DataPlaneStatus, GraphService
from domain.ports.services.pot_management import (
    PotAggregateStatus,
    PotInfo,
    PotManagementService,
    SourceInfo,
)
from domain.ports.services.setup import SetupOrchestrator
from domain.ports.services.skill_manager import (
    AgentTargetPort,
    SkillInfo,
    SkillManager,
    SkillNudge,
    SkillOperationResult,
    SkillStatus,
)

__all__ = [
    "AgentTargetPort",
    "AuthIdentity",
    "AuthService",
    "ConfigService",
    "DataPlaneStatus",
    "GraphService",
    "PotAggregateStatus",
    "PotInfo",
    "PotManagementService",
    "SetupOrchestrator",
    "SkillInfo",
    "SkillManager",
    "SkillNudge",
    "SkillOperationResult",
    "SkillStatus",
    "SourceInfo",
]
