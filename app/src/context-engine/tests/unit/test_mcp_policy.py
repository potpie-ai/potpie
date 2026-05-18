"""Unit tests for embedded MCP policy helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from adapters.inbound.mcp.policy import policy_error_response
from domain.actor import Actor
from domain.ports.policy import (
    ACTION_POT_READ,
    REASON_UNKNOWN_POT,
    RESOURCE_POT,
    PolicyDecision,
)


def test_policy_error_response_none_when_allowed() -> None:
    container = MagicMock()
    container.policy.return_value.authorize.return_value = PolicyDecision(
        allowed=True,
        reason="ok",
        detail="",
        status_code=200,
        metadata={},
    )
    actor = Actor(user_id="u1", surface="mcp", auth_method="api_key")
    assert (
        policy_error_response(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id="pot-1",
        )
        is None
    )


def test_policy_error_response_unknown_pot() -> None:
    container = MagicMock()
    container.policy.return_value.authorize.return_value = PolicyDecision(
        allowed=False,
        reason=REASON_UNKNOWN_POT,
        detail="",
        status_code=404,
        metadata={},
    )
    actor = Actor(user_id="u1", surface="mcp", auth_method="api_key")
    out = policy_error_response(
        container,
        actor=actor,
        resource=RESOURCE_POT,
        action=ACTION_POT_READ,
        pot_id="bad-pot",
    )
    assert out is not None
    assert out["error"] == "policy_denied"
    assert out["reason"] == REASON_UNKNOWN_POT
    assert "Unknown pot_id" in out["detail"]
