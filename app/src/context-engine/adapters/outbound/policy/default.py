"""Default in-process :class:`PolicyPort` adapter.

Wraps the existing settings + pot-resolution + reconciliation-flag checks
behind a single ``authorize`` call. Keeps the route layer thin and turns
"is this allowed?" into a structured decision rather than an ad-hoc chain.
"""

from __future__ import annotations

from typing import Any, Mapping

from domain.actor import Actor
from domain.ports.policy import (
    ACTION_APPLY_WRITE,
    ACTION_CONNECTOR_FETCH,
    ACTION_CONNECTOR_LIST,
    ACTION_POT_INGEST_EPISODE,
    ACTION_POT_MAINTENANCE,
    ACTION_POT_READ,
    ACTION_POT_RECORD,
    ACTION_POT_RESET,
    ACTION_POT_RESOLVE_CONFLICT,
    ACTION_POT_SUBMIT_EVENT,
    REASON_AGENT_PLANNER_DISABLED,
    REASON_CONTEXT_GRAPH_DISABLED,
    REASON_CONTEXT_GRAPH_UNAVAILABLE,
    REASON_EPISODIC_UNAVAILABLE,
    REASON_MAINTENANCE_WRITE_DISABLED,
    REASON_RECONCILIATION_AGENT_UNAVAILABLE,
    REASON_RECONCILIATION_DISABLED,
    REASON_UNKNOWN_POT,
    REASON_UNSUPPORTED_RESOURCE,
    PolicyDecision,
)
from domain.ports.pot_resolution import PotResolutionPort
from domain.ports.settings import ContextEngineSettingsPort
from domain.reconciliation_flags import (
    agent_planner_enabled,
    allow_edge_classify_write_enabled,
    classify_modified_edges_enabled,
    reconciliation_enabled,
)


class DefaultPolicyAdapter:
    """In-process :class:`PolicyPort` over settings + pot resolution.

    Delegated checks (in order, where applicable):

    1. Engine enabled (``settings.is_enabled()``).
    2. For mutating pot actions: reconciliation + agent planner flags + agent
       wired on container.
    3. ``pot_resolution.resolve_pot(pot_id)`` — short-circuit ``unknown_pot``.
    4. For maintenance writes: classify-modified-edges flags.

    The actor argument is forwarded but currently unused — the adapter does
    not yet enforce per-user authorization; hosts that need that compose a
    second adapter in front of this one.
    """

    def __init__(
        self,
        *,
        settings: ContextEngineSettingsPort,
        pots: PotResolutionPort,
        reconciliation_agent_available: bool,
        context_graph_available: bool,
        episodic_available: bool,
    ) -> None:
        self._settings = settings
        self._pots = pots
        self._reconciliation_agent_available = reconciliation_agent_available
        self._context_graph_available = context_graph_available
        self._episodic_available = episodic_available

    def authorize(
        self,
        *,
        actor: Actor | None,
        resource: str,
        action: str,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        del actor  # reserved for future per-actor enforcement
        ctx = dict(context or {})
        if resource == "pot":
            return self._authorize_pot(action, ctx)
        if resource == "connector":
            return self._authorize_connector(action, ctx)
        if resource == "apply":
            return self._authorize_apply(action, ctx)
        return PolicyDecision.deny(
            REASON_UNSUPPORTED_RESOURCE,
            detail=f"Unknown policy resource: {resource}",
            status_code=400,
        )

    # ------------------------------------------------------------------
    # pot.*
    # ------------------------------------------------------------------
    def _authorize_pot(
        self, action: str, ctx: dict[str, Any]
    ) -> PolicyDecision:
        pot_id = ctx.get("pot_id")
        require_reco = action in {
            ACTION_POT_SUBMIT_EVENT,
            ACTION_POT_RECORD,
        }
        require_engine = action != ACTION_POT_READ or bool(pot_id)
        if require_engine and not self._settings.is_enabled():
            return PolicyDecision.deny(
                REASON_CONTEXT_GRAPH_DISABLED,
                detail="Context graph is disabled (CONTEXT_GRAPH_ENABLED).",
            )
        if require_reco:
            if not reconciliation_enabled():
                return PolicyDecision.deny(
                    REASON_RECONCILIATION_DISABLED,
                    detail="Reconciliation is disabled (CONTEXT_ENGINE_RECONCILIATION_ENABLED).",
                )
            if not agent_planner_enabled():
                return PolicyDecision.deny(
                    REASON_AGENT_PLANNER_DISABLED,
                    detail="Agent planner is disabled (CONTEXT_ENGINE_AGENT_PLANNER_ENABLED).",
                )
            if not self._reconciliation_agent_available:
                return PolicyDecision.deny(
                    REASON_RECONCILIATION_AGENT_UNAVAILABLE,
                    detail=(
                        "No reconciliation agent on the container; install "
                        "context-engine[reconciliation-agent] and enable "
                        "CONTEXT_ENGINE_AGENT_PLANNER_ENABLED."
                    ),
                )
        if action == ACTION_POT_RESET and not self._context_graph_available:
            return PolicyDecision.deny(
                REASON_CONTEXT_GRAPH_UNAVAILABLE,
                detail="Unified context graph port is not configured.",
            )
        if action in {ACTION_POT_RESOLVE_CONFLICT} and not self._episodic_available:
            return PolicyDecision.deny(
                REASON_EPISODIC_UNAVAILABLE,
                detail="Episodic graph backend unavailable.",
                status_code=503,
            )
        if action == ACTION_POT_MAINTENANCE:
            dry_run = bool(ctx.get("dry_run", True))
            if not dry_run and not (
                classify_modified_edges_enabled()
                and allow_edge_classify_write_enabled()
            ):
                return PolicyDecision.deny(
                    REASON_MAINTENANCE_WRITE_DISABLED,
                    detail=(
                        "Server must set CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES=1 "
                        "and CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE=1 to apply writes."
                    ),
                    status_code=403,
                )
        if action == ACTION_POT_INGEST_EPISODE and not self._settings.is_enabled():
            return PolicyDecision.deny(
                REASON_CONTEXT_GRAPH_DISABLED,
                detail="Context graph is disabled (CONTEXT_GRAPH_ENABLED).",
            )

        if pot_id is not None:
            resolved = self._pots.resolve_pot(str(pot_id))
            if resolved is None:
                return PolicyDecision.deny(
                    REASON_UNKNOWN_POT,
                    detail=(
                        "Unknown pot_id for this user (create with POST "
                        "/api/v2/context/pots and attach at least one repository)."
                    ),
                    status_code=404,
                )
            return PolicyDecision.allow(resolved_pot_id=resolved.pot_id)

        return PolicyDecision.allow()

    # ------------------------------------------------------------------
    # connector.* / apply.*
    # ------------------------------------------------------------------
    def _authorize_connector(
        self, action: str, ctx: dict[str, Any]
    ) -> PolicyDecision:
        del ctx
        if action in {ACTION_CONNECTOR_FETCH, ACTION_CONNECTOR_LIST}:
            return PolicyDecision.allow()
        return PolicyDecision.deny(
            REASON_UNSUPPORTED_RESOURCE,
            detail=f"Unknown connector action: {action}",
            status_code=400,
        )

    def _authorize_apply(
        self, action: str, ctx: dict[str, Any]
    ) -> PolicyDecision:
        del ctx
        if action != ACTION_APPLY_WRITE:
            return PolicyDecision.deny(
                REASON_UNSUPPORTED_RESOURCE,
                detail=f"Unknown apply action: {action}",
                status_code=400,
            )
        if not self._settings.is_enabled():
            return PolicyDecision.deny(
                REASON_CONTEXT_GRAPH_DISABLED,
                detail="Context graph is disabled (CONTEXT_GRAPH_ENABLED).",
            )
        if not self._context_graph_available:
            return PolicyDecision.deny(
                REASON_CONTEXT_GRAPH_UNAVAILABLE,
                detail="Unified context graph port is not configured.",
            )
        return PolicyDecision.allow()
