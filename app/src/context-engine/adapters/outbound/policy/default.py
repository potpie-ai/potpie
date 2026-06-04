"""Default in-process :class:`PolicyPort` adapter.

Wraps the existing settings + pot-resolution + reconciliation-flag checks
behind a single ``authorize`` call. Keeps the route layer thin and turns
"is this allowed?" into a structured decision rather than an ad-hoc chain.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from domain.actor import Actor
from domain.ports.policy import (
    ACTION_APPLY_WRITE,
    ACTION_CONNECTOR_FETCH,
    ACTION_CONNECTOR_LIST,
    ACTION_POT_INGEST_EPISODE,
    ACTION_POT_READ,
    ACTION_POT_RECORD,
    ACTION_POT_RESET,
    ACTION_POT_RESOLVE_CONFLICT,
    ACTION_POT_SUBMIT_EVENT,
    REASON_AGENT_PLANNER_DISABLED,
    REASON_CONTEXT_GRAPH_DISABLED,
    REASON_CONTEXT_GRAPH_UNAVAILABLE,
    REASON_EPISODIC_UNAVAILABLE,
    REASON_FORBIDDEN,
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
    reconciliation_enabled,
)


# Same dev-only switch the standalone HTTP auth dependency reads (kept as a
# string literal rather than importing the inbound layer to avoid an
# import cycle: container -> policy, deps -> container).
_ALLOW_NO_AUTH_ENV = "CONTEXT_ENGINE_ALLOW_NO_AUTH"

# Surfaces that are stamped server-side only and never settable from a
# client header (see ``_resolve_actor``). A non-actor-scoped resolver is
# acceptable for these because the principal is trusted internal code.
_SERVER_TRUSTED_SURFACES: frozenset[str] = frozenset({"system", "webhook"})
_SERVER_TRUSTED_AUTH: frozenset[str] = frozenset({"system", "webhook_signature"})


def _dev_no_auth() -> bool:
    return os.getenv(_ALLOW_NO_AUTH_ENV, "").strip().lower() in {"1", "true", "yes"}


class DefaultPolicyAdapter:
    """In-process :class:`PolicyPort` over settings + pot resolution.

    Delegated checks (in order, where applicable):

    1. Engine enabled (``settings.is_enabled()``).
    2. For mutating pot actions: reconciliation + agent planner flags + agent
       wired on container.
    3. ``pot_resolution.resolve_pot(pot_id)`` — short-circuit ``unknown_pot``.
    4. For maintenance writes: classify-modified-edges flags.

    Tenant boundary (hard security contract). This adapter does **not**
    itself know which pots an actor owns — that ownership lives in the
    injected :class:`PotResolutionPort`. The contract is therefore:

    * A host serving a network/multi-tenant surface MUST wire an
      actor-scoped resolver (one whose ``resolve_pot`` returns ``None`` for
      pots the caller cannot access) and mark it ``actor_scoped = True``.
      Potpie does this via ``UserScopedContextGraphPotResolution``.
    * When the wired resolver is **not** actor-scoped, every pot-scoped
      action is denied (403) unless the caller is a server-stamped internal
      principal (``system``/``webhook`` surface — never client-assertable)
      or the operator set the loud, dev-only
      ``CONTEXT_ENGINE_ALLOW_NO_AUTH`` escape hatch (single-tenant
      standalone). This makes the module safe-by-default instead of
      relying on an undocumented host wrapper.
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
        ctx = dict(context or {})
        if resource == "pot":
            return self._authorize_pot(action, ctx, actor)
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
    def _tenant_boundary_ok(self, actor: Actor | None) -> bool:
        """Whether a pot-scoped action may proceed for ``actor``.

        Allowed when the wired resolver enforces per-actor scope, or the
        caller is a server-stamped internal principal, or the dev-only
        no-auth escape hatch is set. Otherwise denied — see the class
        docstring's hard security contract.
        """
        if getattr(self._pots, "actor_scoped", False):
            return True
        if _dev_no_auth():
            return True
        surface = getattr(actor, "surface", None)
        auth_method = getattr(actor, "auth_method", None)
        return (
            surface in _SERVER_TRUSTED_SURFACES
            and auth_method in _SERVER_TRUSTED_AUTH
        )

    def _authorize_pot(
        self, action: str, ctx: dict[str, Any], actor: Actor | None = None
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
        if action == ACTION_POT_INGEST_EPISODE and not self._settings.is_enabled():
            return PolicyDecision.deny(
                REASON_CONTEXT_GRAPH_DISABLED,
                detail="Context graph is disabled (CONTEXT_GRAPH_ENABLED).",
            )

        if pot_id is not None:
            if not self._tenant_boundary_ok(actor):
                return PolicyDecision.deny(
                    REASON_FORBIDDEN,
                    detail=(
                        "Per-actor pot authorization is not configured for "
                        "this deployment: the host must wire an actor-scoped "
                        "pot resolver (see DefaultPolicyAdapter contract). "
                        "Refusing pot-scoped access. Set "
                        "CONTEXT_ENGINE_ALLOW_NO_AUTH=1 for single-tenant "
                        "local dev only."
                    ),
                    status_code=403,
                )
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
