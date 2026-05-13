"""Tests for the ``repo_attach`` → ``repository.added`` event bootstrap.

When a repository is attached to a pot, ``add_pot_repository`` calls the
``_emit_bootstrap_event`` helper to submit an event that the
context-engine's batched agent picks up and uses to seed the graph.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.modules.context_graph.attach_repo_to_pot import (
    _emit_bootstrap_event,
)


pytestmark = pytest.mark.unit


def _repo_row(**overrides) -> SimpleNamespace:
    base = dict(
        id="r1",
        pot_id="pot-1",
        owner="acme",
        repo="widgets",
        provider="github",
        provider_host="github.com",
        default_branch="main",
        remote_url="https://github.com/acme/widgets",
        external_repo_id="42",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_emits_repository_added_event_with_expected_fields(monkeypatch) -> None:
    captured = {}

    fake_receipt = SimpleNamespace(event_id="evt-bootstrap-1")

    submission = MagicMock()
    submission.submit.return_value = fake_receipt

    container = MagicMock()
    container.ingestion_submission.return_value = submission

    def _build_container(_db, _uid):
        return container

    monkeypatch.setattr(
        "app.modules.context_graph.wiring.build_container_for_user_session",
        _build_container,
    )

    db = MagicMock()
    eid = _emit_bootstrap_event(
        db,
        pot_id="pot-1",
        repo_row=_repo_row(),
        submitted_by_user_id="user-7",
    )

    assert eid == "evt-bootstrap-1"
    assert submission.submit.call_count == 1
    args, kwargs = submission.submit.call_args
    request = args[0]
    captured["request"] = request

    # The submission goes through the agent_reconciliation pipeline so it
    # ends up in the same debounced batch the dispatcher already polls.
    assert request.ingestion_kind == "agent_reconciliation"
    assert request.source_system == "github"
    assert request.event_type == "repository"
    assert request.action == "added"
    assert request.source_id == "repo_added:acme/widgets"
    assert request.repo_name == "acme/widgets"
    assert request.payload["owner"] == "acme"
    assert request.payload["repo"] == "widgets"
    assert request.payload["default_branch"] == "main"
    assert request.payload["submitted_by_user_id"] == "user-7"
    # Async path — never block on the agent run.
    assert kwargs.get("sync") is False
    assert kwargs.get("wait") is False


def test_returns_none_when_container_build_fails(monkeypatch) -> None:
    """If the container can't be built (e.g. context-engine disabled), don't raise."""
    def _explode(_db, _uid):
        raise RuntimeError("container unavailable")

    monkeypatch.setattr(
        "app.modules.context_graph.wiring.build_container_for_user_session",
        _explode,
    )
    db = MagicMock()
    out = _emit_bootstrap_event(
        db,
        pot_id="pot-1",
        repo_row=_repo_row(),
        submitted_by_user_id="u",
    )
    assert out is None


def test_returns_none_when_submit_raises(monkeypatch) -> None:
    """A submission failure must NOT break the repo-attach response."""
    submission = MagicMock()
    submission.submit.side_effect = RuntimeError("queue down")

    container = MagicMock()
    container.ingestion_submission.return_value = submission
    monkeypatch.setattr(
        "app.modules.context_graph.wiring.build_container_for_user_session",
        lambda _db, _uid: container,
    )
    db = MagicMock()
    out = _emit_bootstrap_event(
        db,
        pot_id="pot-1",
        repo_row=_repo_row(),
        submitted_by_user_id="u",
    )
    assert out is None
