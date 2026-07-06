"""Read-only Linear GraphQL client for CLI workspace, team, and issue listing."""

from __future__ import annotations

from typing import Any

from potpie.cli.auth.credentials_store import (
    get_active_linear_organization_id,
    get_linear_tokens,
    list_linear_organizations,
    set_active_linear_organization,
)
from potpie.cli.auth.errors import CliAuthError
from potpie.cli.auth.http import AuthHttpClient, AuthHttpError, HttpClient
from potpie.cli.auth.integration_profile import linear_workspaces_from_entry
from potpie.cli.auth.integration_session import ensure_valid_integration_tokens

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
_HTTP_TIMEOUT = 30.0

_TEAMS_QUERY = """
query LinearTeams {
  viewer {
    organization {
      id
      name
      urlKey
      teams {
        nodes {
          id
          key
          name
        }
      }
    }
  }
}
"""

_TEAM_ISSUES_QUERY = """
query TeamIssues($teamId: ID!, $first: Int!) {
  issues(
    filter: { team: { id: { eq: $teamId } } }
    first: $first
    orderBy: updatedAt
    includeArchived: false
  ) {
    nodes {
      id
      identifier
      title
      url
      updatedAt
      createdAt
      priority
      state { name }
      team { key name }
      assignee { name }
    }
  }
}
"""


class LinearReadError(CliAuthError):
    """Failed to read Linear data with stored credentials."""


def _access_token(organization_id: str | None = None) -> str:
    org_id = organization_id or get_active_linear_organization_id()
    tokens = (
        get_linear_tokens(org_id)
        if org_id
        else ensure_valid_integration_tokens("linear")
    )
    if org_id and not tokens.get("access_token"):
        tokens = ensure_valid_integration_tokens("linear")
    access_token = str(tokens.get("access_token") or "").strip()
    if not access_token:
        raise LinearReadError("Linear is not connected. Run: potpie linear login")
    return access_token


def load_linear_read_credentials(
    organization_id: str | None = None,
) -> dict[str, Any]:
    """Return stored Linear credentials including workspace preferences."""
    org_id = organization_id or get_active_linear_organization_id()
    tokens = (
        get_linear_tokens(org_id)
        if org_id
        else ensure_valid_integration_tokens("linear")
    )
    if not str(tokens.get("access_token") or "").strip():
        raise LinearReadError("Linear is not connected. Run: potpie linear login")
    return tokens


def _parse_team_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node.get("id"),
        "key": str(node.get("key") or "").strip().upper(),
        "name": node.get("name"),
        "type": "team",
    }


def _parse_issue_node(node: dict[str, Any]) -> dict[str, Any]:
    team_payload = node.get("team")
    team_name = None
    if isinstance(team_payload, dict):
        key = str(team_payload.get("key") or "").strip()
        name = str(team_payload.get("name") or "").strip()
        if key and name:
            team_name = f"{key} ({name})"
        else:
            team_name = key or name or None
    state_payload = node.get("state")
    status = state_payload.get("name") if isinstance(state_payload, dict) else None
    assignee_payload = node.get("assignee")
    assignee = (
        assignee_payload.get("name") if isinstance(assignee_payload, dict) else None
    )
    return {
        "id": node.get("id"),
        "identifier": node.get("identifier"),
        "title": node.get("title"),
        "summary": node.get("title"),
        "status": status,
        "team": team_name,
        "assignee": assignee,
        "priority": node.get("priority"),
        "created": node.get("createdAt"),
        "updated": node.get("updatedAt"),
        "url": node.get("url"),
    }


def _linear_graphql(
    access_token: str,
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    http: HttpClient | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        response = http.post(
            LINEAR_GRAPHQL_URL,
            headers=headers,
            json={"query": query, "variables": variables or {}},
        )
    except AuthHttpError as exc:
        raise LinearReadError(str(exc)) from exc
    finally:
        if owns:
            http.close()
    if response.status_code != 200:
        raise LinearReadError(f"Linear API HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise LinearReadError("Linear API returned non-JSON response") from exc
    if not isinstance(payload, dict):
        raise LinearReadError("Linear API returned an unexpected response body")
    errors = payload.get("errors")
    if errors:
        raise LinearReadError("Linear API rejected the request")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise LinearReadError("Linear API returned no data")
    return data


def fetch_linear_workspaces(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return Linear workspaces (organizations) connected to this CLI."""
    rows: list[dict[str, Any]] = []
    for entry in list_linear_organizations()[: max(1, min(int(limit), 50))]:
        rows.append(
            {
                "id": entry.get("id"),
                "key": entry.get("key"),
                "name": entry.get("name"),
                "type": "workspace",
                "active": bool(entry.get("active")),
            }
        )
    return rows


def fetch_linear_teams(
    *,
    organization_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return teams within a connected Linear workspace."""
    org_id = organization_id or get_active_linear_organization_id()
    if not org_id:
        raise LinearReadError("No active Linear workspace. Run: potpie linear select")
    access_token = _access_token(org_id)
    data = _linear_graphql(access_token, _TEAMS_QUERY)
    viewer = data.get("viewer")
    org = viewer.get("organization") if isinstance(viewer, dict) else None
    if isinstance(org, dict) and str(org.get("id") or "") != str(org_id):
        raise LinearReadError(
            "Linear token does not match the selected workspace. "
            "Run: potpie linear login --add"
        )
    teams_conn = org.get("teams") if isinstance(org, dict) else None
    nodes = teams_conn.get("nodes") if isinstance(teams_conn, dict) else None
    if not isinstance(nodes, list):
        return []
    rows: list[dict[str, Any]] = []
    for node in nodes:
        if isinstance(node, dict):
            rows.append(_parse_team_node(node))
    return rows[: max(1, min(int(limit), 50))]


def fetch_linear_issues_in_team(
    team_id: str,
    *,
    organization_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent issues for a Linear team."""
    team_id_value = str(team_id or "").strip()
    if not team_id_value:
        raise LinearReadError("Linear team id is required.")
    org_id = organization_id or get_active_linear_organization_id()
    access_token = _access_token(org_id)
    data = _linear_graphql(
        access_token,
        _TEAM_ISSUES_QUERY,
        {
            "teamId": team_id_value,
            "first": max(1, min(int(limit), 50)),
        },
    )
    conn = data.get("issues")
    nodes = conn.get("nodes") if isinstance(conn, dict) else None
    if not isinstance(nodes, list):
        return []
    rows: list[dict[str, Any]] = []
    for node in nodes:
        if isinstance(node, dict):
            rows.append(_parse_issue_node(node))
    return rows


def _normalize_org_ref(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def resolve_linear_organization(
    workspaces: list[dict[str, Any]],
    *,
    org_key: str | None = None,
    organization_id: str | None = None,
    credentials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a workspace dict from key, id, saved prefs, or interactive default."""
    creds = credentials or {}
    prefs = linear_workspaces_from_entry(creds)
    explicit_key = str(org_key or "").strip()
    explicit_id = str(organization_id or "").strip()
    if explicit_key or explicit_id:
        ref = _normalize_org_ref(explicit_key or explicit_id)
        org_id = explicit_id
    else:
        ref = _normalize_org_ref(
            prefs.get("linear_organization_key")
            or prefs.get("linear_organization_id")
            or ""
        )
        org_id = str(prefs.get("linear_organization_id") or "").strip()
    if org_id:
        match = next((row for row in workspaces if str(row.get("id")) == org_id), None)
        if match:
            return match
    if ref:
        for row in workspaces:
            candidates = {
                _normalize_org_ref(str(row.get("key") or "")),
                _normalize_org_ref(str(row.get("name") or "")),
                _normalize_org_ref(str(row.get("id") or "")),
            }
            if ref in candidates:
                return row
        if org_id:
            return {"id": org_id, "key": org_key or org_id, "name": org_key or org_id}
        return {"id": "", "key": org_key or ref, "name": org_key or ref}
    active = next((row for row in workspaces if row.get("active")), None)
    if active:
        return active
    if len(workspaces) == 1:
        return workspaces[0]
    if workspaces:
        return workspaces[0]
    raise LinearReadError("No Linear workspaces connected. Run: potpie linear login")


def resolve_linear_team(
    teams: list[dict[str, Any]],
    *,
    team_key: str | None = None,
    team_id: str | None = None,
    credentials: dict[str, Any] | None = None,
    organization_id: str | None = None,
) -> dict[str, Any]:
    """Resolve a team dict from key, id, saved prefs, or the first available team."""
    creds = credentials or {}
    prefs = linear_workspaces_from_entry(creds)
    saved_org = str(prefs.get("linear_organization_id") or "").strip()
    if organization_id and saved_org and saved_org != organization_id:
        prefs = {}
    explicit_key = str(team_key or "").strip().upper()
    explicit_id = str(team_id or "").strip()
    if explicit_key:
        key = explicit_key
        tid = explicit_id
    else:
        key = str(prefs.get("linear_team") or "").strip().upper()
        tid = str(explicit_id or prefs.get("linear_team_id") or "").strip()
    if key:
        match = next((team for team in teams if team.get("key") == key), None)
        if match:
            return match
        if tid:
            return {"id": tid, "key": key, "name": key}
        return {"id": "", "key": key, "name": key}
    if tid:
        match = next((team for team in teams if team.get("id") == tid), None)
        if match:
            return match
        return {"id": tid, "key": tid, "name": tid}
    if teams:
        return teams[0]
    raise LinearReadError("No Linear teams found for this workspace.")


def activate_linear_organization(org_id: str) -> None:
    """Switch the active Linear workspace for subsequent CLI reads."""
    set_active_linear_organization(org_id)
