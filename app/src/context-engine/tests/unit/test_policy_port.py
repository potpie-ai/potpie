"""DefaultPolicyAdapter — single decision call site for engine policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from adapters.outbound.policy import DefaultPolicyAdapter
from domain.ports.policy import (
    ACTION_APPLY_WRITE,
    ACTION_POT_INGEST_EPISODE,
    ACTION_POT_MAINTENANCE,
    ACTION_POT_READ,
    ACTION_POT_RECORD,
    ACTION_POT_RESET,
    ACTION_POT_SUBMIT_EVENT,
    REASON_AGENT_PLANNER_DISABLED,
    REASON_CONTEXT_GRAPH_DISABLED,
    REASON_MAINTENANCE_WRITE_DISABLED,
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
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = mapping or {}

    def resolve_pot(self, pot_id: str):
        if pot_id in self._mapping:
            return _ResolvedPot(pot_id=pot_id)
        return None


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


def test_pot_record_denied_when_agent_unavailable():
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


def test_pot_maintenance_blocks_writes_without_flags():
    with patch.dict(
        os.environ,
        {
            "CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES": "0",
            "CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE": "0",
        },
    ):
        decision = _adapter(pots={"p1": "x"}).authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_MAINTENANCE,
            context={"pot_id": "p1", "dry_run": False},
        )
    assert not decision.allowed
    assert decision.reason == REASON_MAINTENANCE_WRITE_DISABLED
    assert decision.status_code == 403


def test_pot_maintenance_dry_run_allowed_without_flags():
    with patch.dict(
        os.environ,
        {
            "CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES": "0",
            "CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE": "0",
        },
    ):
        decision = _adapter(pots={"p1": "x"}).authorize(
            actor=None,
            resource=RESOURCE_POT,
            action=ACTION_POT_MAINTENANCE,
            context={"pot_id": "p1", "dry_run": True},
        )
    assert decision.allowed


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
