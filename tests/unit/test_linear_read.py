"""Unit tests for interactive Linear read flows."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from potpie.cli.auth import linear_read as lr
from potpie.auth.adapters.linear_read_client import LinearReadError


@dataclass
class LinearReadStub:
    """Record the flow's read requests while retaining real selection logic."""

    workspaces: list[dict[str, Any]]
    teams_by_org: dict[str, list[dict[str, Any]]]
    credentials_by_org: dict[str | None, dict[str, Any]]
    issues_by_request: dict[tuple[str, str | None, int], list[dict[str, Any]]]
    activated: list[str] = field(default_factory=list)
    team_requests: list[str | None] = field(default_factory=list)
    issue_requests: list[tuple[str, str | None, int]] = field(default_factory=list)
    saved: list[dict[str, Any]] = field(default_factory=list)

    def load_credentials(self, organization_id: str | None = None) -> dict[str, Any]:
        assert organization_id in self.credentials_by_org, (
            f"Unexpected credentials request for organization {organization_id!r}"
        )
        return self.credentials_by_org[organization_id]

    def fetch_workspaces(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.workspaces[:limit]

    def activate_organization(self, org_id: str) -> None:
        assert org_id in self.credentials_by_org, (
            f"Unexpected organization activation: {org_id!r}"
        )
        self.activated.append(org_id)

    def fetch_teams(
        self, *, organization_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        self.team_requests.append(organization_id)
        assert organization_id in self.teams_by_org, (
            f"Unexpected teams request for organization {organization_id!r}"
        )
        return self.teams_by_org[organization_id][:limit]

    def fetch_issues(
        self,
        team_id: str,
        *,
        organization_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        request = (team_id, organization_id, limit)
        self.issue_requests.append(request)
        assert request in self.issues_by_request, (
            f"Unexpected issue request (team, organization, limit): {request!r}"
        )
        return self.issues_by_request[request]

    def save_prefs(
        self,
        *,
        organization_id: str,
        organization_key: str,
        team_key: str,
        team_id: str,
    ) -> None:
        self.saved.append(
            {
                "organization_id": organization_id,
                "organization_key": organization_key,
                "team_key": team_key,
                "team_id": team_id,
            }
        )


@pytest.fixture()
def linear_reads(monkeypatch: pytest.MonkeyPatch) -> LinearReadStub:
    """The read flow has no http= seam, so record its adapter-boundary calls."""
    stub = LinearReadStub(
        workspaces=[{"id": "org-1", "key": "acme", "name": "Acme"}],
        teams_by_org={
            "org-1": [{"id": "t1", "key": "ENG", "name": "Engineering"}],
        },
        credentials_by_org={
            None: {"access_token": "initial-token"},
            "org-1": {"access_token": "acme-token"},
        },
        issues_by_request={
            ("t1", "org-1", 10): [{"identifier": "ENG-1", "title": "Ship it"}],
        },
    )
    monkeypatch.setattr(lr, "load_linear_read_credentials", stub.load_credentials)
    monkeypatch.setattr(lr, "fetch_linear_workspaces", stub.fetch_workspaces)
    monkeypatch.setattr(lr, "activate_linear_organization", stub.activate_organization)
    monkeypatch.setattr(lr, "fetch_linear_teams", stub.fetch_teams)
    monkeypatch.setattr(lr, "fetch_linear_issues_in_team", stub.fetch_issues)
    monkeypatch.setattr(lr, "save_linear_workspace_prefs", stub.save_prefs)
    return stub


def assert_selection(
    reads: LinearReadStub,
    *,
    organization_id: str = "org-1",
    organization_key: str = "acme",
    team_id: str = "t1",
    team_key: str = "ENG",
    limit: int = 10,
) -> None:
    assert reads.activated == [organization_id]
    assert reads.team_requests == [organization_id]
    assert reads.issue_requests == [(team_id, organization_id, limit)]
    assert reads.saved == [
        {
            "organization_id": organization_id,
            "organization_key": organization_key,
            "team_key": team_key,
            "team_id": team_id,
        }
    ]


def test_run_linear_use_flow_requires_tty_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(LinearReadError, match="Interactive workspace selection"):
        lr.run_linear_use_flow()


def test_run_linear_use_flow_no_workspaces(
    monkeypatch: pytest.MonkeyPatch, linear_reads: LinearReadStub
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    linear_reads.workspaces = []
    with pytest.raises(LinearReadError, match="No Linear workspaces connected"):
        lr.run_linear_use_flow()
    assert linear_reads.team_requests == []
    assert linear_reads.issue_requests == []


def test_run_linear_use_flow_single_workspace_auto_pick(
    monkeypatch: pytest.MonkeyPatch, linear_reads: LinearReadStub
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    teams = linear_reads.teams_by_org["org-1"]
    prompt = MagicMock(return_value=teams[0])
    monkeypatch.setattr(lr, "_prompt_workspace", prompt)
    linear_reads.issues_by_request = {
        ("t1", "org-1", 5): [{"identifier": "ENG-1", "title": "Ship it"}]
    }

    result = lr.run_linear_use_flow(limit=5)

    assert result["workspace_key"] == "acme"
    assert result["team_key"] == "ENG"
    assert result["items"][0]["identifier"] == "ENG-1"
    prompt.assert_called_once_with(teams, label="Linear team")
    assert_selection(linear_reads, limit=5)


def test_run_linear_use_flow_non_interactive_org_and_team(
    monkeypatch: pytest.MonkeyPatch, linear_reads: LinearReadStub
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    # Neither requested selection is the first item or the saved preference.
    linear_reads.workspaces.insert(
        0, {"id": "org-other", "key": "other", "name": "Other", "active": True}
    )
    linear_reads.teams_by_org["org-1"].insert(
        0, {"id": "t-other", "key": "QA", "name": "Quality"}
    )
    linear_reads.credentials_by_org[None]["workspaces"] = {
        "linear_organization_id": "org-other"
    }
    linear_reads.credentials_by_org["org-1"]["workspaces"] = {
        "linear_organization_id": "org-1",
        "linear_team": "QA",
        "linear_team_id": "t-other",
    }
    prompt = MagicMock(side_effect=AssertionError("Unexpected interactive prompt"))
    monkeypatch.setattr(lr, "_prompt_workspace", prompt)

    result = lr.run_linear_use_flow(org_key="acme", team_key="eng")

    assert result["product"] == "linear"
    assert result["workspace_key"] == "acme"
    assert result["team_key"] == "ENG"
    prompt.assert_not_called()
    assert_selection(linear_reads)


def test_run_linear_use_flow_non_interactive_org_only(
    monkeypatch: pytest.MonkeyPatch, linear_reads: LinearReadStub
) -> None:
    """Org-only runs select the saved team without prompting."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    linear_reads.teams_by_org["org-1"].insert(
        0, {"id": "t-other", "key": "QA", "name": "Quality"}
    )
    linear_reads.credentials_by_org["org-1"]["workspaces"] = {
        "linear_organization_id": "org-1",
        "linear_team": "ENG",
        "linear_team_id": "t1",
    }
    prompt = MagicMock(side_effect=AssertionError("Unexpected interactive prompt"))
    monkeypatch.setattr(lr, "_prompt_workspace", prompt)

    result = lr.run_linear_use_flow(org_key="acme")

    assert result["team_key"] == "ENG"
    prompt.assert_not_called()
    assert_selection(linear_reads)


def test_run_linear_use_flow_missing_org_id(
    monkeypatch: pytest.MonkeyPatch, linear_reads: LinearReadStub
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(LinearReadError, match="Could not resolve Linear workspace"):
        lr.run_linear_use_flow(org_key="bad", team_key="ENG")
    assert linear_reads.activated == []
    assert linear_reads.team_requests == []
    assert linear_reads.issue_requests == []
    assert linear_reads.saved == []


def test_run_linear_use_flow_no_teams(
    monkeypatch: pytest.MonkeyPatch, linear_reads: LinearReadStub
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    linear_reads.teams_by_org["org-1"] = []
    with pytest.raises(LinearReadError, match="No Linear teams found"):
        lr.run_linear_use_flow(org_key="acme")
    assert linear_reads.team_requests == ["org-1"]
    assert linear_reads.issue_requests == []
    assert linear_reads.saved == []


def test_run_linear_use_flow_multi_workspace_prompts(
    monkeypatch: pytest.MonkeyPatch, linear_reads: LinearReadStub
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    organization = {"id": "org-2", "key": "b", "name": "B"}
    team = {"id": "t2", "key": "OPS", "name": "Operations"}
    linear_reads.workspaces.append(organization)
    linear_reads.credentials_by_org["org-2"] = {"access_token": "b-token"}
    linear_reads.teams_by_org["org-2"] = [
        {"id": "t-other", "key": "QA", "name": "Quality"},
        team,
    ]
    linear_reads.issues_by_request = {("t2", "org-2", 10): [{"identifier": "OPS-2"}]}
    prompt = MagicMock(side_effect=[organization, team])
    monkeypatch.setattr(lr, "_prompt_workspace", prompt)

    result = lr.run_linear_use_flow()

    assert result["workspace_key"] == "b"
    assert result["team_key"] == "OPS"
    assert result["items"][0]["identifier"] == "OPS-2"
    prompt.assert_any_call(linear_reads.workspaces, label="Linear workspace")
    prompt.assert_any_call(linear_reads.teams_by_org["org-2"], label="Linear team")
    assert prompt.call_count == 2
    assert_selection(
        linear_reads,
        organization_id="org-2",
        organization_key="b",
        team_id="t2",
        team_key="OPS",
    )
