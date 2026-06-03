"""Optional construction of reconciliation agents (env + optional deps)."""

from __future__ import annotations

import logging

from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_tools import ReconciliationToolsPort
from domain.reconciliation_flags import agent_planner_enabled

logger = logging.getLogger(__name__)


def try_pydantic_deep_reconciliation_agent(
    *,
    tools: ReconciliationToolsPort | None = None,
) -> ReconciliationAgentPort | None:
    """When the agent planner flag allows it (default: on), return a pydantic-deep agent if installed.

    Pass ``tools`` to give the agent a bounded read-only context-tool surface
    (see ``ContextGraphReconciliationTools``).
    """
    if not agent_planner_enabled():
        return None
    try:
        from adapters.outbound.reconciliation.pydantic_deep_agent import (
            PydanticDeepReconciliationAgent,
        )

        return PydanticDeepReconciliationAgent(tools=tools)
    except ImportError:
        logger.warning(
            "Agent planner is enabled but pydantic-deep is not installed; "
            "install context-engine[reconciliation-agent]"
        )
        return None
