"""DefaultPolicyAdapter — single decision call site for engine policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from context_engine.adapters.outbound.policy import DefaultPolicyAdapter
from context_engine.domain.actor import Actor
from context_engine.domain.ports.policy import (
    ACTION_APPLY_WRITE,
    ACTION_POT_INGEST_EPISODE,
    ACTION_POT_READ,
    ACTION_POT_RECORD,
    ACTION_POT_RESET,
    ACTION_POT_SUBMIT_EVENT,
    REASON_AGENT_PLANNER_DISABLED,
    REASON_CONTEXT_GRAPH_DISABLED,
    REASON_FORBIDDEN,
    REASON_RECONCILIATION_AGENT_UNAVAILABLE,
    REASON_RECONCILIATION_DISABLED,
    REASON_UNKNOWN_POT,
    REASON_UNSUPPORTED_RESOURCE,
    RESOURCE_APPLY,
    RESOURCE_POT,
)


class _Settings:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def neo4j_uri(self) -> None:
        return None

    def neo4j_user(self) -> None:
        return None

    def neo4j_password(self) -> None:
        return None

    def backfill_max_prs_per_run(self) -> int:
        return 100


@dataclass
class _ResolvedPot:
    pot_id: str


class _Pots:
    """Actor-scoped resolver stand-in (mirrors the production host wiring:
    ``resolve_pot`` only returns pots the caller may access)."""

    actor_scoped = True

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = mapping or {}

    def resolve_pot(self, pot_id: str):
        if pot_id in self._mapping:
            return _ResolvedPot(pot_id=pot_id)
        return None


class _WidePots(_Pots):
    """Non-actor-scoped resolver (standalone env-map / worker style)."""

    actor_scoped = False


def _adapter(
    *,
    enabled: bool = True,
    pots: dict[str, str] | None = None,
    agent_available: bool = True,
    context_graph_available: bool = True,
    episodic_available: bool = True,
) -> DefaultPolicyAdapter:
    return DefaultPolicyAdapter(
        settings=_Settings(enabled),
        pots=_Pots(pots),
        reconciliation_agent_available=agent_available,
        context_graph_available=context_graph_available,
        episodic_available=episodic_available,
    )


def test_pot_read_allows_known_pot():
    decision = _adapter(pots={"p1": "owner/repo"}).authorize(
        actor=None,
        resource=RESOURCE_POT,
        action=ACTION_POT_READ,
        context={"pot_id": "p1"},
    )
    assert decision.allowed
    assert decision.metadata["resolved_pot_id"] == "p1"


def test_pot_read_denies_unknown_pot_with_404():
    decision = _adapter(pots={"p1": "owner/repo"}).authorize(
        actor=None,
        resource=RESOURCE_POT,
        action=ACTION_POT_READ,
        context={"pot_id": "missing"},
    )
    assert not decision.allowed
    assert decision.reason == REASON_UNKNOWN_POT
    assert decision.status_code == 404


def test_pot_read_without_pot_skips_engine_check():
    """Listing-style endpoints with no pot still pass when no pot_id is given."""
    decision = _adapter(enabled=False).authorize(
        actor=None,
        resource=RESOURCE_POT,
        action=ACTION_POT_READ,
    )
    assert decision.allowed


def test_pot_submit_event_chains_settings_and_flags():
    decision = _adapter(enabled=False, pots={"p1": "x"}).authorize(
        actor=None,
        resource=RESOURCE_POT,
        action=ACTION_POT_SUBMIT_EVENT,
        context={"pot_id": "p1"},
    )
    assert not decision.allowed
    assert decision.reason == REASON_CONTEXT_GRAPH_DISABLED


def test_pot_submit_event_denied_when_reconciliation_disabled():
    with patch.dict(os.environ, {"CONTEXT_ENGINE_RECONCILIATION_ENABLED": "0"}):
        decision = _adapter(pots={"p1": "x"}).authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_SUBMIT_EVENT,
            context={"pot_id": "p1"},
        )
    assert not decision.allowed
    assert decision.reason == REASON_RECONCILIATION_DISABLED


def test_pot_submit_event_denied_when_planner_disabled():
    with patch.dict(os.environ, {"CONTEXT_ENGINE_AGENT_PLANNER_ENABLED": "0"}):
        decision = _adapter(pots={"p1": "x"}).authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_SUBMIT_EVENT,
            context={"pot_id": "p1"},
        )
    assert not decision.allowed
    assert decision.reason == REASON_AGENT_PLANNER_DISABLED


def test_pot_record_denied_by_default_planner_is_opt_in():
    """Step 11: the LLM planner is opt-in, so a known-pot record is denied by
    default (no env override) — canonical writes use the local semantic path."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CONTEXT_ENGINE_AGENT_PLANNER_ENABLED", None)
        decision = _adapter(pots={"p1": "x"}).authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_RECORD,
            context={"pot_id": "p1"},
        )
    assert not decision.allowed
    assert decision.reason == REASON_AGENT_PLANNER_DISABLED


def test_pot_record_denied_when_agent_unavailable():
    # The agent-availability gate is only reachable once the (now opt-in)
    # planner flag is enabled; otherwise the planner-disabled gate fires first.
    with patch.dict(os.environ, {"CONTEXT_ENGINE_AGENT_PLANNER_ENABLED": "1"}):
        decision = _adapter(pots={"p1": "x"}, agent_available=False).authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_RECORD,
            context={"pot_id": "p1"},
        )
    assert not decision.allowed
    assert decision.reason == REASON_RECONCILIATION_AGENT_UNAVAILABLE


def test_pot_ingest_episode_engine_disabled():
    decision = _adapter(enabled=False, pots={"p1": "x"}).authorize(
        actor=None,
        resource=RESOURCE_POT,
        action=ACTION_POT_INGEST_EPISODE,
        context={"pot_id": "p1"},
    )
    assert not decision.allowed
    assert decision.reason == REASON_CONTEXT_GRAPH_DISABLED


def test_pot_reset_denies_unknown_pot():
    decision = _adapter(pots={"p1": "x"}).authorize(
        actor=None,
        resource=RESOURCE_POT,
        action=ACTION_POT_RESET,
        context={"pot_id": "ghost"},
    )
    assert not decision.allowed
    assert decision.reason == REASON_UNKNOWN_POT


def test_apply_write_requires_engine_enabled():
    decision = _adapter(enabled=False).authorize(
        actor=None,
        resource=RESOURCE_APPLY,
        action=ACTION_APPLY_WRITE,
    )
    assert not decision.allowed
    assert decision.reason == REASON_CONTEXT_GRAPH_DISABLED


def test_unsupported_resource_returns_400():
    decision = _adapter().authorize(
        actor=None,
        resource="bogus",
        action="read",
    )
    assert not decision.allowed
    assert decision.reason == REASON_UNSUPPORTED_RESOURCE
    assert decision.status_code == 400


@pytest.mark.parametrize(
    "action",
    [
        ACTION_POT_READ,
        ACTION_POT_SUBMIT_EVENT,
        ACTION_POT_RECORD,
        ACTION_POT_RESET,
        ACTION_POT_INGEST_EPISODE,
    ],
)
def test_known_pot_id_short_circuits_when_unknown(action):
    decision = _adapter(pots={"p1": "x"}).authorize(
        actor=None,
        resource=RESOURCE_POT,
        action=action,
        context={"pot_id": "ghost"},
    )
    assert not decision.allowed
    assert decision.reason == REASON_UNKNOWN_POT


# --- Tenant-boundary contract (C-1) -----------------------------------------


def _wide_adapter() -> DefaultPolicyAdapter:
    return DefaultPolicyAdapter(
        settings=_Settings(True),
        pots=_WidePots({"p1": "owner/repo"}),
        reconciliation_agent_available=True,
        context_graph_available=True,
        episodic_available=True,
    )


def test_pot_scoped_denied_when_resolver_not_actor_scoped():
    """A non-actor-scoped resolver must not grant pot access by default."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CONTEXT_ENGINE_ALLOW_NO_AUTH", None)
        decision = _wide_adapter().authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            context={"pot_id": "p1"},
        )
    assert not decision.allowed
    assert decision.reason == REASON_FORBIDDEN
    assert decision.status_code == 403


def test_pot_scoped_allowed_with_dev_escape_hatch():
    with patch.dict(os.environ, {"CONTEXT_ENGINE_ALLOW_NO_AUTH": "1"}):
        decision = _wide_adapter().authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            context={"pot_id": "p1"},
        )
    assert decision.allowed
    assert decision.metadata["resolved_pot_id"] == "p1"


def test_pot_scoped_allowed_for_server_trusted_internal_actor():
    """Workers / signature-verified webhooks may use the wide resolver."""
    system_actor = Actor(user_id="system", surface="system", auth_method="system")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CONTEXT_ENGINE_ALLOW_NO_AUTH", None)
        decision = _wide_adapter().authorize(
            actor=system_actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            context={"pot_id": "p1"},
        )
    assert decision.allowed


def test_pot_scoped_denied_for_spoofed_http_actor_on_wide_resolver():
    """An http/cli/mcp caller can never substitute for actor scoping."""
    http_actor = Actor(user_id="attacker", surface="http", auth_method="api_key")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CONTEXT_ENGINE_ALLOW_NO_AUTH", None)
        decision = _wide_adapter().authorize(
            actor=http_actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            context={"pot_id": "p1"},
        )
    assert not decision.allowed
    assert decision.reason == REASON_FORBIDDEN
