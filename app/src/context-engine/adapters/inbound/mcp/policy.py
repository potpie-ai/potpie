"""Policy checks for MCP tools (mirrors HTTP router _enforce)."""

from __future__ import annotations

from typing import Any

from bootstrap.container import ContextEngineContainer
from domain.ports.policy import REASON_UNKNOWN_POT

UNKNOWN_POT_DETAIL = (
    "Unknown pot_id for this user (create with POST /api/v2/context/pots "
    "and attach at least one repository)."
)


def policy_error_response(
    container: ContextEngineContainer,
    *,
    actor: Any,
    resource: str,
    action: str,
    pot_id: str,
) -> dict[str, Any] | None:
    """Return an error dict when denied; None when allowed."""
    decision = container.policy().authorize(
        actor=actor,
        resource=resource,
        action=action,
        context={"pot_id": pot_id},
    )
    if decision.allowed:
        return None
    detail = (
        UNKNOWN_POT_DETAIL
        if decision.reason == REASON_UNKNOWN_POT
        else decision.detail or decision.reason
    )
    return {
        "ok": False,
        "error": "policy_denied",
        "reason": decision.reason,
        "status_code": decision.status_code,
        "detail": detail,
    }
