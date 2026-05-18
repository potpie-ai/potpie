"""Unit tests for the Linear backfill use case.

Backfill enumerates issues for a Linear team, synthesises a webhook
envelope per issue, and submits each through the context-engine
ingestion service. Tests fake the SQLAlchemy session, the Linear
adapter, and the context-engine container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from integrations.application import backfill_linear_source as bf_module
from integrations.application.backfill_linear_source import backfill_linear_source


# ----- fakes -----------------------------------------------------------


@dataclass
class _FakeIntegration:
    integration_id: str = "int-1"
    integration_type: str = "linear"
    active: bool = True
    unique_identifier: str = "org-1"
    auth_data: dict | None = None
    scope_data: dict | None = None

    def __post_init__(self):
        if self.auth_data is None:
            self.auth_data = {"access_token": "encrypted-token"}
        if self.scope_data is None:
            self.scope_data = {"org_id": "org-1"}


@dataclass
class _FakePotSource:
    id: str = "ps-1"
    pot_id: str = "pot-1"
    integration_id: str | None = "int-1"
    provider: str = "linear"
    source_kind: str = "issue_tracker_team"
    scope_json: Any = None
    sync_enabled: bool = True

    def __post_init__(self):
        if self.scope_json is None:
            self.scope_json = {"team_id": "team-1"}


class _FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self, by_model=None):
        self.by_model = by_model or {}
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        return _FakeQuery(self.by_model.get(model, []))

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@dataclass
class _Ref:
    id: str
    identifier: str = "ENG-1"
    updated_at: Any = None


class _FakeAdapter:
    def __init__(self, issues: list[dict], iter_error: Exception | None = None):
        self._issues = issues
        self._iter_error = iter_error
        self.iter_called_with = None
        self.get_called_with: list[str] = []

    def iter_issues(self, *, scope, updated_after=None):
        self.iter_called_with = scope
        if self._iter_error is not None:
            raise self._iter_error
        for i, issue in enumerate(self._issues):
            yield _Ref(id=issue["id"], identifier=issue.get("identifier") or str(i))

    def get_issue(self, *, scope, issue_id):
        self.get_called_with.append(issue_id)
        for issue in self._issues:
            if issue["id"] == issue_id:
                return issue
        raise LookupError(issue_id)


class _FakeConnector:
    def __init__(self, fail_for: set[str] | None = None):
        self.fail_for = fail_for or set()
        self.normalized: list[bytes] = []

    def normalize_webhook(self, body, headers):
        self.normalized.append(body)
        import json

        envelope = json.loads(body.decode("utf-8"))
        issue = envelope["data"]
        if issue["id"] in self.fail_for:
            return None  # connector chose to ignore this issue

        class _Event:
            def __init__(self, identifier):
                self.event_id = f"ev-{identifier}"
                self.source_system = "linear"
                self.event_type = "linear_issue"
                self.action = "create"
                self.source_id = f"linear:issue:{identifier}:create"
                self.source_event_id = None
                self.repo_name = "team:team-1"
                self.provider = "linear"
                self.provider_host = "linear.app"
                self.payload = {"action": "create", "issue": issue}

        return _Event(issue.get("identifier") or issue["id"])


class _FakeReceipt:
    def __init__(self, event_id="ev", status="queued", duplicate=False):
        self.event_id = event_id
        self.status = status
        self.duplicate = duplicate


class _FakeSubmission:
    def __init__(self, fail_for: set[str] | None = None):
        self.submitted: list = []
        self.fail_for = fail_for or set()

    def submit(self, request):
        self.submitted.append(request)
        if request.source_id in self.fail_for:
            raise RuntimeError("simulated submission failure")
        return _FakeReceipt(event_id=request.source_id)


class _FakeSettings:
    def __init__(self, enabled=True):
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled


class _FakeConnectors:
    def __init__(self, connector):
        self._connector = connector

    def find_for_webhook(self, kind):
        return self._connector if kind == "linear" else None


class _FakeContainer:
    def __init__(self, *, connector, submission, enabled=True):
        self.connectors = _FakeConnectors(connector)
        self.settings = _FakeSettings(enabled=enabled)
        self._submission = submission

    def ingestion_submission(self, db):
        return self._submission


# ----- patch helpers ---------------------------------------------------


def _patch_all(
    monkeypatch,
    *,
    integration_rows,
    source_rows,
    container,
    adapter,
    decrypt_value="plain-token",
):
    from app.modules.context_graph.context_graph_pot_source_model import (
        ContextGraphPotSource,
    )
    from integrations.adapters.outbound.postgres.integration_model import Integration

    db = _FakeSession(
        by_model={Integration: integration_rows, ContextGraphPotSource: source_rows}
    )
    monkeypatch.setattr(bf_module, "_build_container", lambda d: container)
    monkeypatch.setattr(
        bf_module, "LinearIssueTrackerAdapter", lambda token: adapter
    )
    monkeypatch.setattr(bf_module, "decrypt_token", lambda t: decrypt_value)
    monkeypatch.setattr(bf_module, "touch_pot_source_sync", lambda *a, **kw: None)
    return db


# ----- tests -----------------------------------------------------------


def test_returns_skipped_when_project_source_missing(monkeypatch):
    db = _patch_all(
        monkeypatch,
        integration_rows=[],
        source_rows=[],
        container=_FakeContainer(connector=_FakeConnector(), submission=_FakeSubmission()),
        adapter=_FakeAdapter([]),
    )
    result = backfill_linear_source(db, "missing")
    assert result["status"] == "skipped"
    assert result["reason"] == "not_found_or_not_linear"


def test_returns_error_when_team_id_missing(monkeypatch):
    source = _FakePotSource(scope_json={})
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[source],
        container=_FakeContainer(connector=_FakeConnector(), submission=_FakeSubmission()),
        adapter=_FakeAdapter([]),
    )
    result = backfill_linear_source(db, source.id)
    assert result["status"] == "error"
    assert result["reason"] == "missing_team_id"


def test_returns_error_when_integration_id_missing(monkeypatch):
    source = _FakePotSource(integration_id=None)
    db = _patch_all(
        monkeypatch,
        integration_rows=[],
        source_rows=[source],
        container=_FakeContainer(connector=_FakeConnector(), submission=_FakeSubmission()),
        adapter=_FakeAdapter([]),
    )
    result = backfill_linear_source(db, source.id)
    assert result["status"] == "error"
    assert result["reason"] == "missing_integration_id"


def test_returns_error_when_integration_not_found(monkeypatch):
    source = _FakePotSource()
    db = _patch_all(
        monkeypatch,
        integration_rows=[],  # no integration rows
        source_rows=[source],
        container=_FakeContainer(connector=_FakeConnector(), submission=_FakeSubmission()),
        adapter=_FakeAdapter([]),
    )
    result = backfill_linear_source(db, source.id)
    assert result["status"] == "error"
    assert result["reason"] == "integration_not_found"


def test_returns_skipped_when_engine_disabled(monkeypatch):
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[_FakePotSource()],
        container=_FakeContainer(
            connector=_FakeConnector(),
            submission=_FakeSubmission(),
            enabled=False,
        ),
        adapter=_FakeAdapter([]),
    )
    result = backfill_linear_source(db, "ps-1")
    assert result["status"] == "skipped"
    assert result["reason"] == "context_engine_disabled"


def test_happy_path_ingests_each_issue(monkeypatch):
    issues = [
        {"id": "i-1", "identifier": "ENG-1", "title": "First"},
        {"id": "i-2", "identifier": "ENG-2", "title": "Second"},
        {"id": "i-3", "identifier": "ENG-3", "title": "Third"},
    ]
    submission = _FakeSubmission()
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[_FakePotSource()],
        container=_FakeContainer(
            connector=_FakeConnector(), submission=submission
        ),
        adapter=_FakeAdapter(issues),
    )
    result = backfill_linear_source(db, "ps-1")
    assert result["status"] == "success"
    assert result["ingested"] == 3
    assert result["failed"] == 0
    submitted_source_ids = {r.source_id for r in submission.submitted}
    assert submitted_source_ids == {
        "linear:issue:ENG-1:create",
        "linear:issue:ENG-2:create",
        "linear:issue:ENG-3:create",
    }
    # Every submission carries integration_id + is_backfill in payload
    for req in submission.submitted:
        assert req.payload.get("integration_id") == "int-1"
        assert req.payload.get("is_backfill") is True
        assert req.source_channel == "backfill"


def test_max_issues_caps_enumeration(monkeypatch):
    issues = [{"id": f"i-{i}", "identifier": f"E-{i}"} for i in range(20)]
    submission = _FakeSubmission()
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[_FakePotSource()],
        container=_FakeContainer(
            connector=_FakeConnector(), submission=submission
        ),
        adapter=_FakeAdapter(issues),
    )
    result = backfill_linear_source(db, "ps-1", max_issues=5)
    assert result["ingested"] == 5
    assert len(submission.submitted) == 5


def test_partial_failure_continues(monkeypatch):
    issues = [{"id": f"i-{i}", "identifier": f"E-{i}"} for i in range(5)]
    submission = _FakeSubmission(fail_for={"linear:issue:E-2:create"})
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[_FakePotSource()],
        container=_FakeContainer(
            connector=_FakeConnector(), submission=submission
        ),
        adapter=_FakeAdapter(issues),
    )
    result = backfill_linear_source(db, "ps-1")
    assert result["status"] == "partial_success"
    assert result["ingested"] == 4
    assert result["failed"] == 1
    assert "E-2" in result["errors"][0]


def test_enumerator_error_marked_as_run_error(monkeypatch):
    submission = _FakeSubmission()
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[_FakePotSource()],
        container=_FakeContainer(
            connector=_FakeConnector(), submission=submission
        ),
        adapter=_FakeAdapter([], iter_error=RuntimeError("linear 401")),
    )
    result = backfill_linear_source(db, "ps-1")
    assert result["status"] == "error"
    assert "linear 401" in (result["reason"] or "")
    assert submission.submitted == []


def test_token_decrypt_failure(monkeypatch):
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[_FakePotSource()],
        container=_FakeContainer(
            connector=_FakeConnector(), submission=_FakeSubmission()
        ),
        adapter=_FakeAdapter([]),
        decrypt_value=None,
    )

    # decrypt_token is patched to return None already; emulate the
    # ``access_token=None`` post-decrypt scenario by setting it to None
    # via the decrypt patch
    def _raises(t):
        raise RuntimeError("decrypt boom")

    import integrations.application.backfill_linear_source as mod

    mod.decrypt_token = _raises  # type: ignore[assignment]
    result = backfill_linear_source(db, "ps-1")
    assert result["status"] == "error"
    assert result["reason"] == "missing_access_token"


def test_connector_skip_treated_as_failure(monkeypatch):
    """If the connector returns None for a synthesised envelope, count as failed."""
    issues = [
        {"id": "i-1", "identifier": "ENG-1"},
        {"id": "i-2", "identifier": "ENG-2"},
    ]
    connector = _FakeConnector(fail_for={"i-1"})
    submission = _FakeSubmission()
    db = _patch_all(
        monkeypatch,
        integration_rows=[_FakeIntegration()],
        source_rows=[_FakePotSource()],
        container=_FakeContainer(connector=connector, submission=submission),
        adapter=_FakeAdapter(issues),
    )
    result = backfill_linear_source(db, "ps-1")
    assert result["ingested"] == 1
    assert result["failed"] == 1
    assert "ENG-1" in result["errors"][0]
