"""Interactive Linear read flows (workspace + team pickers and issue fetch)."""

from __future__ import annotations

import sys
from typing import Any

from potpie.cli.auth.atlassian_read import _prompt_workspace
from adapters.outbound.cli_auth.credentials_store import save_linear_workspace_prefs
from adapters.outbound.cli_auth.linear_read_client import (
    LinearReadError,
    activate_linear_organization,
    fetch_linear_issues_in_team,
    fetch_linear_teams,
    fetch_linear_workspaces,
    load_linear_read_credentials,
    resolve_linear_organization,
    resolve_linear_team,
)


def run_linear_use_flow(
    *,
    org_key: str | None = None,
    team_key: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Pick a Linear workspace and team, persist choice, and fetch issues."""
    if not sys.stdin.isatty() and not (org_key or team_key):
        raise LinearReadError(
            "Interactive workspace selection requires a terminal. "
            "Use: potpie linear select --org potpie --team ENG"
        )
    ctx = load_linear_read_credentials()
    workspaces = fetch_linear_workspaces()
    if not workspaces:
        raise LinearReadError(
            "No Linear workspaces connected. Run: potpie linear login"
        )

    default_org = str(org_key or "").strip()
    if default_org or (not sys.stdin.isatty() and (org_key or team_key)):
        picked_org = resolve_linear_organization(
            workspaces,
            org_key=default_org or None,
            credentials=ctx,
        )
    elif len(workspaces) == 1:
        picked_org = workspaces[0]
    else:
        picked_org = _prompt_workspace(workspaces, label="Linear workspace")

    org_id = str(picked_org.get("id") or "").strip()
    if not org_id:
        raise LinearReadError(
            f"Could not resolve Linear workspace {picked_org.get('key')!r}. "
            "Run: potpie linear login --add"
        )
    activate_linear_organization(org_id)
    ctx = load_linear_read_credentials(organization_id=org_id)

    teams = fetch_linear_teams(organization_id=org_id)
    default_team = str(team_key or "").strip().upper()
    if team_key or default_team or not sys.stdin.isatty():
        picked_team = resolve_linear_team(
            teams,
            team_key=default_team or None,
            credentials=ctx,
            organization_id=org_id,
        )
    elif teams and sys.stdin.isatty():
        picked_team = _prompt_workspace(teams, label="Linear team")
    else:
        raise LinearReadError("No Linear teams found for this workspace.")

    team_key_value = str(picked_team.get("key") or "").strip().upper()
    team_id = str(picked_team.get("id") or "").strip()
    if not team_id:
        raise LinearReadError(
            f"Could not resolve Linear team {team_key_value!r}. Run: potpie linear ls"
        )
    rows = fetch_linear_issues_in_team(
        team_id,
        organization_id=org_id,
        limit=limit,
    )
    save_linear_workspace_prefs(
        organization_id=org_id,
        organization_key=str(picked_org.get("key") or ""),
        team_key=team_key_value,
        team_id=team_id,
    )
    return {
        "product": "linear",
        "workspace_key": picked_org.get("key"),
        "workspace_name": picked_org.get("name"),
        "team_key": team_key_value,
        "team_name": picked_team.get("name"),
        "items": rows,
    }
