"""Unit tests for Linear read-only GraphQL client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.outbound.cli_auth.http import AuthHttpError
from adapters.outbound.cli_auth import linear_read_client as lrc


def test_fetch_linear_workspaces_from_organizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lrc,
        "list_linear_organizations",
        lambda: [
            {"id": "org-1", "key": "acme", "name": "Acme", "active": True},
            {"id": "org-2", "key": "beta", "name": "Beta", "active": False},
        ],
    )
    rows = lrc.fetch_linear_workspaces()
    assert len(rows) == 2
    assert rows[0]["key"] == "acme"
    assert rows[0]["type"] == "workspace"
    assert rows[0]["active"] is True


def test_load_linear_read_credentials_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lrc, "get_active_linear_organization_id", lambda: "org-1")
    monkeypatch.setattr(lrc, "get_linear_tokens", lambda _org: {"access_token": ""})
    with pytest.raises(lrc.LinearReadError, match="not connected"):
        lrc.load_linear_read_credentials()


def test_access_token_falls_back_to_ensure_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lrc, "get_active_linear_organization_id", lambda: "org-1")
    monkeypatch.setattr(
        lrc,
        "get_linear_tokens",
        lambda _org: {"access_token": ""},
    )
    monkeypatch.setattr(
        lrc,
        "ensure_valid_integration_tokens",
        lambda _provider: {"access_token": "fallback-token"},
    )
    assert lrc._access_token("org-1") == "fallback-token"


def test_linear_graphql_http_error() -> None:
    client = MagicMock()
    client.post.side_effect = AuthHttpError("offline")
    with pytest.raises(lrc.LinearReadError, match="offline"):
        lrc._linear_graphql("tok", "query {}", http=client)


def test_linear_graphql_non_200() -> None:
    response = MagicMock()
    response.status_code = 503
    client = MagicMock()
    client.post.return_value = response
    with pytest.raises(lrc.LinearReadError, match="HTTP 503"):
        lrc._linear_graphql("tok", "query {}", http=client)


def test_linear_graphql_bad_json() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    client = MagicMock()
    client.post.return_value = response
    with pytest.raises(lrc.LinearReadError, match="non-JSON"):
        lrc._linear_graphql("tok", "query {}", http=client)


def test_linear_graphql_graphql_errors() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"errors": [{"message": "bad token"}]}
    client = MagicMock()
    client.post.return_value = response
    with pytest.raises(lrc.LinearReadError, match="rejected"):
        lrc._linear_graphql("tok", "query {}", http=client)


def test_linear_graphql_missing_data() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": None}
    client = MagicMock()
    client.post.return_value = response
    with pytest.raises(lrc.LinearReadError, match="no data"):
        lrc._linear_graphql("tok", "query {}", http=client)


def test_fetch_linear_teams_requires_active_org(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lrc, "get_active_linear_organization_id", lambda: None)
    with pytest.raises(lrc.LinearReadError, match="No active Linear workspace"):
        lrc.fetch_linear_teams()


def test_fetch_linear_teams_org_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lrc, "get_active_linear_organization_id", lambda: "org-1")
    monkeypatch.setattr(lrc, "_access_token", lambda _org: "tok")
    monkeypatch.setattr(
        lrc,
        "_linear_graphql",
        lambda _tok, _query, **_k: {
            "viewer": {
                "organization": {
                    "id": "other-org",
                    "teams": {"nodes": []},
                }
            }
        },
    )
    with pytest.raises(lrc.LinearReadError, match="does not match"):
        lrc.fetch_linear_teams(organization_id="org-1")


def test_fetch_linear_teams_parses_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lrc, "get_active_linear_organization_id", lambda: "org-1")
    monkeypatch.setattr(lrc, "_access_token", lambda _org: "tok")
    monkeypatch.setattr(
        lrc,
        "_linear_graphql",
        lambda _tok, _query, **_k: {
            "viewer": {
                "organization": {
                    "id": "org-1",
                    "teams": {
                        "nodes": [
                            {"id": "t1", "key": "eng", "name": "Engineering"},
                        ]
                    },
                }
            }
        },
    )
    teams = lrc.fetch_linear_teams(organization_id="org-1")
    assert teams[0]["key"] == "ENG"
    assert teams[0]["name"] == "Engineering"


def test_fetch_linear_issues_in_team_requires_team_id() -> None:
    with pytest.raises(lrc.LinearReadError, match="team id is required"):
        lrc.fetch_linear_issues_in_team("")


def test_fetch_linear_issues_in_team_parses_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lrc, "get_active_linear_organization_id", lambda: "org-1")
    monkeypatch.setattr(lrc, "_access_token", lambda _org: "tok")
    monkeypatch.setattr(
        lrc,
        "_linear_graphql",
        lambda _tok, _query, _vars, **_k: {
            "issues": {
                "nodes": [
                    {
                        "id": "i1",
                        "identifier": "ENG-1",
                        "title": "Fix bug",
                        "url": "https://linear.app/ENG-1",
                        "updatedAt": "2026-01-01",
                        "createdAt": "2025-12-01",
                        "priority": 2,
                        "state": {"name": "In Progress"},
                        "team": {"key": "ENG", "name": "Engineering"},
                        "assignee": {"name": "Ada"},
                    }
                ]
            }
        },
    )
    rows = lrc.fetch_linear_issues_in_team("team-1", organization_id="org-1")
    assert rows[0]["identifier"] == "ENG-1"
    assert rows[0]["status"] == "In Progress"
    assert rows[0]["team"] == "ENG (Engineering)"
    assert rows[0]["assignee"] == "Ada"


def test_resolve_linear_organization_by_key() -> None:
    workspaces = [
        {"id": "org-1", "key": "potpie-ai", "name": "Potpie AI", "active": False},
    ]
    match = lrc.resolve_linear_organization(workspaces, org_key="potpie-ai")
    assert match["id"] == "org-1"


def test_resolve_linear_organization_explicit_key_ignores_saved_prefs() -> None:
    workspaces = [
        {"id": "org-1", "key": "potpie-ai", "name": "Potpie AI", "active": False},
        {"id": "org-2", "key": "other", "name": "Other", "active": True},
    ]
    creds = {
        "linear_organization_id": "org-2",
        "linear_organization_key": "other",
    }
    match = lrc.resolve_linear_organization(
        workspaces,
        org_key="potpie-ai",
        credentials=creds,
    )
    assert match["id"] == "org-1"


def test_resolve_linear_organization_prefers_active() -> None:
    workspaces = [
        {"id": "org-1", "key": "a", "name": "A", "active": False},
        {"id": "org-2", "key": "b", "name": "B", "active": True},
    ]
    match = lrc.resolve_linear_organization(workspaces)
    assert match["id"] == "org-2"


def test_resolve_linear_organization_empty_raises() -> None:
    with pytest.raises(lrc.LinearReadError, match="No Linear workspaces"):
        lrc.resolve_linear_organization([])


def test_resolve_linear_team_by_key() -> None:
    teams = [{"id": "t1", "key": "ENG", "name": "Engineering"}]
    match = lrc.resolve_linear_team(teams, team_key="ENG")
    assert match["id"] == "t1"


def test_resolve_linear_team_unknown_key_returns_stub() -> None:
    teams: list[dict] = []
    match = lrc.resolve_linear_team(teams, team_key="ENG", team_id="t9")
    assert match == {"id": "t9", "key": "ENG", "name": "ENG"}


def test_resolve_linear_team_explicit_key_ignores_saved_team_id() -> None:
    teams: list[dict] = []
    creds = {"linear_team_id": "stale-t1", "linear_team": "OLD"}
    match = lrc.resolve_linear_team(teams, team_key="ENG", credentials=creds)
    assert match == {"id": "", "key": "ENG", "name": "ENG"}


def test_resolve_linear_team_clears_prefs_on_org_mismatch() -> None:
    teams = [{"id": "t1", "key": "ENG", "name": "Engineering"}]
    creds = {
        "linear_organization_id": "other-org",
        "linear_team": "ENG",
        "linear_team_id": "t1",
    }
    match = lrc.resolve_linear_team(
        teams,
        credentials=creds,
        organization_id="org-1",
    )
    assert match["key"] == "ENG"


def test_activate_linear_organization(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        lrc,
        "set_active_linear_organization",
        lambda org_id: called.append(org_id),
    )
    lrc.activate_linear_organization("org-99")
    assert called == ["org-99"]
