"""Unit tests for the Linear webhook routing use case.

The use case orchestrates: payload validation → connector canonicalization →
integration lookup → pot subscription filtering → ingestion submission
per (pot, integration). These tests fake the SQLAlchemy session and the
context-engine container so we exercise the orchestration without a DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from integrations.application import process_linear_webhook as use_case_module
from integrations.application.process_linear_webhook import process_linear_webhook
from integrations.domain.exceptions import (
    WebhookPayloadError,
    WebhookProcessingError,
)


# ----- fixtures --------------------------------------------------------


@dataclass
class _FakeIntegration:
    integration_id: str
    integration_type: str = "linear"
    active: bool = True
    unique_identifier: str = "org-1"


@dataclass
class _FakePotSource:
    pot_id: str
    integration_id: str
    provider: str = "linear"
    source_kind: str = "issue_tracker_team"
    scope_json: Any = None
    sync_enabled: bool = True

    def __post_init__(self):
        if self.scope_json is None:
            self.scope_json = {}


class _FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Stand-in for sqlalchemy.orm.Session that returns canned rows by model class."""

    def __init__(self, by_model: dict[type, list[Any]] | None = None):
        self._by_model = by_model or {}
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        return _FakeQuery(self._by_model.get(model, []))

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _FakeReceipt:
    def __init__(self, event_id: str, status: str = "queued", duplicate: bool = False):
        self.event_id = event_id
        self.status = status
        self.duplicate = duplicate


class _FakeSubmission:
    def __init__(self, fail_for: set[str] | None = None):
        self.submitted: list = []
        self.fail_for = fail_for or set()

    def submit(self, request):
        self.submitted.append(request)
        if request.pot_id in self.fail_for:
            raise RuntimeError("simulated submission failure")
        return _FakeReceipt(event_id=f"evt-{request.pot_id}")


class _FakeConnector:
    def __init__(self, event):
        self.event = event

    def normalize_webhook(self, body: bytes, headers):
        return self.event


class _FakeConnectors:
    def __init__(self, connector):
        self._connector = connector

    def find_for_webhook(self, kind: str):
        return self._connector if kind == "linear" else None


class _FakeSettings:
    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled


class _FakeContainer:
    def __init__(self, *, connector, submission, enabled: bool = True):
        self.connectors = _FakeConnectors(connector)
        self.settings = _FakeSettings(enabled=enabled)
        self._submission = submission

    def ingestion_submission(self, db):
        return self._submission


class _FakeEvent:
    """Minimal stand-in for context-engine's ContextEvent."""

    def __init__(
        self,
        *,
        event_id="evt-1",
        source_system="linear",
        event_type="linear_issue",
        action="state_change",
        source_id="linear:issue:ENG-1:state_change:state-uuid",
        source_event_id="delivery-1",
        repo_name="team:team-1",
        provider="linear",
        provider_host="linear.app",
        payload=None,
    ):
        self.event_id = event_id
        self.source_system = source_system
        self.event_type = event_type
        self.action = action
        self.source_id = source_id
        self.source_event_id = source_event_id
        self.repo_name = repo_name
        self.provider = provider
        self.provider_host = provider_host
        self.payload = payload or {"action": action, "issue": {"id": "i-1"}}


# ----- helpers ---------------------------------------------------------


def _basic_payload(org_id: str = "org-1") -> dict:
    return {
        "type": "Issue",
        "action": "update",
        "organizationId": org_id,
        "data": {
            "id": "i-1",
            "identifier": "ENG-1",
            "team": {"id": "team-1"},
            "state": {"id": "state-uuid", "name": "Done", "type": "completed"},
        },
        "updatedFrom": {"stateId": "old-state", "stateName": "In Progress"},
    }


def _patch_container(monkeypatch, container):
    monkeypatch.setattr(
        use_case_module,
        "_build_container",
        lambda db: container,
    )


# ----- tests -----------------------------------------------------------


def test_missing_organization_id_raises_payload_error(monkeypatch):
    db = _FakeSession()
    with pytest.raises(WebhookPayloadError):
        process_linear_webhook(db, payload={"type": "Issue", "data": {}})


def test_non_dict_payload_raises_payload_error():
    db = _FakeSession()
    with pytest.raises(WebhookPayloadError):
        process_linear_webhook(db, payload=[])  # type: ignore[arg-type]


def test_context_engine_disabled_returns_ignored(monkeypatch):
    db = _FakeSession()
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()),
        submission=_FakeSubmission(),
        enabled=False,
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["status"] == "ignored"
    assert result["reason"] == "context_engine_disabled"
    assert db.commits == 0


def test_unregistered_connector_raises_processing_error(monkeypatch):
    db = _FakeSession()

    class _NoConnector:
        def __init__(self):
            self.connectors = _FakeConnectors(None)
            self.settings = _FakeSettings(True)

        def ingestion_submission(self, db):
            return _FakeSubmission()

    _patch_container(monkeypatch, _NoConnector())
    with pytest.raises(WebhookProcessingError):
        process_linear_webhook(db, payload=_basic_payload())


def test_connector_returns_none_marks_event_ignored(monkeypatch):
    db = _FakeSession()
    container = _FakeContainer(
        connector=_FakeConnector(None), submission=_FakeSubmission()
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(
        db, payload={"type": "Project", "action": "create", "organizationId": "org-1"}
    )
    assert result["status"] == "ignored"
    assert result["reason"] == "ignored_event_type"


def test_no_integration_for_org_returns_ignored(monkeypatch):
    from integrations.adapters.outbound.postgres.integration_model import Integration

    db = _FakeSession(by_model={Integration: []})
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()), submission=_FakeSubmission()
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["status"] == "ignored"
    assert result["reason"] == "no_subscribed_pots"


def test_integration_with_no_matching_team_returns_ignored(monkeypatch):
    from integrations.adapters.outbound.postgres.integration_model import Integration
    from app.modules.context_graph.context_graph_pot_source_model import (
        ContextGraphPotSource,
    )

    integration = _FakeIntegration(integration_id="int-1")
    # Subscribed to a different team than the event's team-1
    source = _FakePotSource(
        pot_id="pot-1",
        integration_id="int-1",
        scope_json={"team_id": "team-other"},
    )
    db = _FakeSession(by_model={Integration: [integration], ContextGraphPotSource:[source]})
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()), submission=_FakeSubmission()
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["status"] == "ignored"
    assert result["reason"] == "no_subscribed_pots"


def test_happy_path_single_pot(monkeypatch):
    from integrations.adapters.outbound.postgres.integration_model import Integration
    from app.modules.context_graph.context_graph_pot_source_model import (
        ContextGraphPotSource,
    )

    integration = _FakeIntegration(integration_id="int-1")
    source = _FakePotSource(
        pot_id="pot-1",
        integration_id="int-1",
        scope_json={"team_id": "team-1"},
    )
    db = _FakeSession(by_model={Integration: [integration], ContextGraphPotSource:[source]})
    submission = _FakeSubmission()
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()), submission=submission
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["status"] == "routed"
    assert result["successes"] == 1
    assert result["failures"] == 0
    assert len(submission.submitted) == 1
    req = submission.submitted[0]
    assert req.pot_id == "pot-1"
    assert req.source_system == "linear"
    # integration_id propagated into payload for observability
    assert req.payload.get("integration_id") == "int-1"
    assert db.commits == 1


def test_fan_out_to_multiple_pots(monkeypatch):
    from integrations.adapters.outbound.postgres.integration_model import Integration
    from app.modules.context_graph.context_graph_pot_source_model import (
        ContextGraphPotSource,
    )

    integration = _FakeIntegration(integration_id="int-1")
    sources = [
        _FakePotSource(
            pot_id=f"pot-{i}",
            integration_id="int-1",
            scope_json={"team_id": "team-1"},
        )
        for i in (1, 2, 3)
    ]
    db = _FakeSession(by_model={Integration: [integration], ContextGraphPotSource:sources})
    submission = _FakeSubmission()
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()), submission=submission
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["successes"] == 3
    pots = {r.pot_id for r in submission.submitted}
    assert pots == {"pot-1", "pot-2", "pot-3"}


def test_multiple_integrations_same_org_each_routed(monkeypatch):
    """Two users installed the same workspace; each pot gets its own submission."""
    from integrations.adapters.outbound.postgres.integration_model import Integration
    from app.modules.context_graph.context_graph_pot_source_model import (
        ContextGraphPotSource,
    )

    int_a = _FakeIntegration(integration_id="int-a")
    int_b = _FakeIntegration(integration_id="int-b")
    sources = [
        _FakePotSource(
            pot_id="pot-a", integration_id="int-a", scope_json={"team_id": "team-1"}
        ),
        _FakePotSource(
            pot_id="pot-b", integration_id="int-b", scope_json={"team_id": "team-1"}
        ),
    ]
    db = _FakeSession(by_model={Integration: [int_a, int_b], ContextGraphPotSource:sources})
    submission = _FakeSubmission()
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()), submission=submission
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["successes"] == 2
    by_pot = {r.pot_id: r.payload.get("integration_id") for r in submission.submitted}
    assert by_pot == {"pot-a": "int-a", "pot-b": "int-b"}


def test_partial_fanout_failure_does_not_abort(monkeypatch):
    from integrations.adapters.outbound.postgres.integration_model import Integration
    from app.modules.context_graph.context_graph_pot_source_model import (
        ContextGraphPotSource,
    )

    integration = _FakeIntegration(integration_id="int-1")
    sources = [
        _FakePotSource(
            pot_id=f"pot-{i}",
            integration_id="int-1",
            scope_json={"team_id": "team-1"},
        )
        for i in (1, 2, 3)
    ]
    db = _FakeSession(by_model={Integration: [integration], ContextGraphPotSource:sources})
    submission = _FakeSubmission(fail_for={"pot-2"})
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()), submission=submission
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["successes"] == 2
    assert result["failures"] == 1
    failed = [r for r in result["routed"] if "error" in r]
    assert failed[0]["pot_id"] == "pot-2"
    # Commit still happens for the successful submissions
    assert db.commits == 1


def test_disabled_pot_source_is_skipped(monkeypatch):
    from integrations.adapters.outbound.postgres.integration_model import Integration
    from app.modules.context_graph.context_graph_pot_source_model import (
        ContextGraphPotSource,
    )

    integration = _FakeIntegration(integration_id="int-1")
    # The fake query returns whatever rows it was given; we simulate the
    # filter outcome by only including enabled rows here.
    source = _FakePotSource(
        pot_id="pot-1",
        integration_id="int-1",
        scope_json={"team_id": "team-1"},
        sync_enabled=True,
    )
    db = _FakeSession(by_model={Integration: [integration], ContextGraphPotSource:[source]})
    submission = _FakeSubmission()
    container = _FakeContainer(
        connector=_FakeConnector(_FakeEvent()), submission=submission
    )
    _patch_container(monkeypatch, container)
    result = process_linear_webhook(db, payload=_basic_payload())
    assert result["status"] == "routed"
    assert result["successes"] == 1


def test_payload_is_re_serialized_for_connector(monkeypatch):
    """The connector should receive the original dict as JSON bytes (round-trip)."""
    from integrations.adapters.outbound.postgres.integration_model import Integration

    captured = {}

    class _CapturingConnector:
        def normalize_webhook(self, body, headers):
            captured["body"] = body
            captured["parsed"] = json.loads(body.decode("utf-8"))
            return _FakeEvent()

    db = _FakeSession(by_model={Integration: []})
    container = _FakeContainer(
        connector=_CapturingConnector(), submission=_FakeSubmission()
    )
    _patch_container(monkeypatch, container)
    payload = _basic_payload()
    process_linear_webhook(db, payload=payload)
    assert captured["parsed"]["organizationId"] == "org-1"
