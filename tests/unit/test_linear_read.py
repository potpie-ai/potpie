"""Unit tests for interactive Linear read flows."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from potpie.cli.auth import linear_read as lr
from adapters.outbound.cli_auth.linear_read_client import LinearReadError


def test_run_linear_use_flow_requires_tty_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(LinearReadError, match="Interactive workspace selection"):
        lr.run_linear_use_flow()


def test_run_linear_use_flow_no_workspaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        lr, "load_linear_read_credentials", lambda **_: {"access_token": "t"}
    )
    monkeypatch.setattr(lr, "fetch_linear_workspaces", lambda: [])
    with pytest.raises(LinearReadError, match="No Linear workspaces connected"):
        lr.run_linear_use_flow()


def test_run_linear_use_flow_single_workspace_auto_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    saved: list[dict] = []

    monkeypatch.setattr(
        lr, "load_linear_read_credentials", lambda **_: {"access_token": "t"}
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_workspaces",
        lambda: [{"id": "org-1", "key": "acme", "name": "Acme"}],
    )
    monkeypatch.setattr(lr, "activate_linear_organization", lambda _org: None)
    monkeypatch.setattr(
        lr,
        "fetch_linear_teams",
        lambda **_: [{"id": "t1", "key": "ENG", "name": "Engineering"}],
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_issues_in_team",
        lambda *_a, **_k: [{"identifier": "ENG-1", "title": "Ship it"}],
    )
    monkeypatch.setattr(
        lr,
        "save_linear_workspace_prefs",
        lambda **kwargs: saved.append(kwargs),
    )
    monkeypatch.setattr(
        lr,
        "_prompt_workspace",
        lambda teams, **_: teams[0],
    )

    result = lr.run_linear_use_flow(limit=5)

    assert result["workspace_key"] == "acme"
    assert result["team_key"] == "ENG"
    assert result["items"][0]["identifier"] == "ENG-1"
    assert saved[0]["organization_id"] == "org-1"
    assert saved[0]["team_key"] == "ENG"


def test_run_linear_use_flow_non_interactive_org_and_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        lr, "load_linear_read_credentials", lambda **_: {"access_token": "t"}
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_workspaces",
        lambda: [{"id": "org-1", "key": "acme", "name": "Acme"}],
    )
    monkeypatch.setattr(lr, "activate_linear_organization", lambda _org: None)
    monkeypatch.setattr(
        lr,
        "resolve_linear_organization",
        lambda workspaces, **_: workspaces[0],
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_teams",
        lambda **_: [{"id": "t1", "key": "ENG", "name": "Engineering"}],
    )
    monkeypatch.setattr(
        lr,
        "resolve_linear_team",
        lambda teams, **_: teams[0],
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_issues_in_team",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(lr, "save_linear_workspace_prefs", lambda **_k: None)

    result = lr.run_linear_use_flow(org_key="acme", team_key="eng")

    assert result["product"] == "linear"
    assert result["team_key"] == "ENG"


def test_run_linear_use_flow_non_interactive_org_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Org-only non-interactive runs must not call the team prompt."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        lr, "load_linear_read_credentials", lambda **_: {"access_token": "t"}
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_workspaces",
        lambda: [{"id": "org-1", "key": "acme", "name": "Acme"}],
    )
    monkeypatch.setattr(lr, "activate_linear_organization", lambda _org: None)
    monkeypatch.setattr(
        lr,
        "resolve_linear_organization",
        lambda workspaces, **_: workspaces[0],
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_teams",
        lambda **_: [{"id": "t1", "key": "ENG", "name": "Engineering"}],
    )
    resolve_team = MagicMock(
        return_value={"id": "t1", "key": "ENG", "name": "Engineering"}
    )
    monkeypatch.setattr(lr, "resolve_linear_team", resolve_team)
    prompt = MagicMock()
    monkeypatch.setattr(lr, "_prompt_workspace", prompt)
    monkeypatch.setattr(
        lr,
        "fetch_linear_issues_in_team",
        lambda *_a, **_k: [{"identifier": "ENG-1"}],
    )
    monkeypatch.setattr(lr, "save_linear_workspace_prefs", lambda **_k: None)

    result = lr.run_linear_use_flow(org_key="acme")

    assert result["team_key"] == "ENG"
    resolve_team.assert_called_once()
    prompt.assert_not_called()


def test_run_linear_use_flow_missing_org_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(lr, "load_linear_read_credentials", lambda **_: {})
    monkeypatch.setattr(
        lr,
        "fetch_linear_workspaces",
        lambda: [{"id": "", "key": "bad", "name": "Bad"}],
    )
    monkeypatch.setattr(
        lr,
        "resolve_linear_organization",
        lambda workspaces, **_: workspaces[0],
    )
    with pytest.raises(LinearReadError, match="Could not resolve Linear workspace"):
        lr.run_linear_use_flow(org_key="bad", team_key="ENG")


def test_run_linear_use_flow_no_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(lr, "load_linear_read_credentials", lambda **_: {})
    monkeypatch.setattr(
        lr,
        "fetch_linear_workspaces",
        lambda: [{"id": "org-1", "key": "acme", "name": "Acme"}],
    )
    monkeypatch.setattr(lr, "activate_linear_organization", lambda _org: None)
    monkeypatch.setattr(
        lr,
        "resolve_linear_organization",
        lambda workspaces, **_: workspaces[0],
    )
    monkeypatch.setattr(lr, "fetch_linear_teams", lambda **_: [])
    with pytest.raises(LinearReadError, match="No Linear teams found"):
        lr.run_linear_use_flow(org_key="acme")


def test_run_linear_use_flow_multi_workspace_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(lr, "load_linear_read_credentials", lambda **_: {})
    monkeypatch.setattr(
        lr,
        "fetch_linear_workspaces",
        lambda: [
            {"id": "org-1", "key": "a", "name": "A"},
            {"id": "org-2", "key": "b", "name": "B"},
        ],
    )
    monkeypatch.setattr(lr, "activate_linear_organization", lambda _org: None)
    monkeypatch.setattr(
        lr,
        "fetch_linear_teams",
        lambda **_: [{"id": "t1", "key": "ENG", "name": "Engineering"}],
    )
    monkeypatch.setattr(
        lr,
        "_prompt_workspace",
        MagicMock(
            side_effect=[
                {"id": "org-2", "key": "b", "name": "B"},
                {"id": "t1", "key": "ENG", "name": "Engineering"},
            ]
        ),
    )
    monkeypatch.setattr(
        lr,
        "fetch_linear_issues_in_team",
        lambda *_a, **_k: [{"identifier": "ENG-2"}],
    )
    monkeypatch.setattr(lr, "save_linear_workspace_prefs", lambda **_k: None)

    result = lr.run_linear_use_flow()

    assert result["workspace_key"] == "b"
    assert result["items"][0]["identifier"] == "ENG-2"
