"""Optional construction of reconciliation agents (env + optional deps)."""

from __future__ import annotations

import logging

from potpie_context_engine.domain.ports.reconciliation_agent import (
    ReconciliationAgentPort,
)
from potpie_context_engine.domain.ports.reconciliation_tools import (
    ReconciliationToolsPort,
)
from potpie_context_core.reconciliation_config import ReconciliationConfig
from potpie_context_core.reconciliation_flags import reconciliation_config_from_env

logger = logging.getLogger(__name__)


def try_pydantic_deep_reconciliation_agent(
    *,
    tools: ReconciliationToolsPort | None = None,
    reconciliation_config: ReconciliationConfig | None = None,
) -> ReconciliationAgentPort | None:
    """Return the optional agent when the injected planner policy allows it.

    Pass ``tools`` to give the agent a bounded read-only context-tool surface
    (see ``ContextGraphReconciliationTools``).
    """
    config = reconciliation_config or reconciliation_config_from_env()
    if not config.agent_planner_enabled:
        return None
    try:
        from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
            PydanticDeepReconciliationAgent,
        )

        return PydanticDeepReconciliationAgent(tools=tools)
    except ImportError:
        logger.warning(
            "Agent planner is enabled but pydantic-deep is not installed; "
            "install context-engine[reconciliation-agent]"
        )
        return None
