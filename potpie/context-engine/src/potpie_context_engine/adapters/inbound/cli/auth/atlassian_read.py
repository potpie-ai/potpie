"""Interactive Atlassian read flows (Jira project / Confluence space pickers).

The HTTP/data layer (fetchers, parsers, credential loaders) lives in
``potpie_context_engine.adapters.outbound.cli_auth.atlassian_read_client``; this inbound module owns the
interactive workspace-selection flow (prompts, output, terminal guards) and
re-exports the client symbols for callers that historically imported them here.
"""

from __future__ import annotations

import sys
from typing import Any

import typer

from potpie_context_engine.adapters.outbound.cli_auth.atlassian_read_client import (  # noqa: F401  (re-export)
    AtlassianReadError,
    _auth_header_variants,
    _cloud_id_from_credentials,
    _get_json,
    _post_json,
    _site_url_from_credentials,
    fetch_confluence_content_sample,
    fetch_confluence_pages_in_space,
    fetch_confluence_spaces_sample,
    fetch_jira_issues_in_project,
    fetch_jira_issues_sample,
    fetch_jira_projects,
    load_atlassian_read_credentials,
    load_confluence_read_credentials,
    load_jira_read_credentials,
)
from potpie_context_engine.adapters.outbound.cli_auth.credentials_store import (
    save_confluence_workspace_prefs,
    save_jira_workspace_prefs,
)
from potpie_context_engine.adapters.inbound.cli.ui.output import print_plain_line


def _prompt_choice(label: str, *, default: str = "1") -> int:
    while True:
        choice = typer.prompt(label, default=default).strip()
        try:
            return int(choice)
        except ValueError:
            print_plain_line("Enter a number from the list.", as_json=False)


def _prompt_workspace(
    items: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    if not items:
        raise AtlassianReadError(f"No {label} found for this account.")
    print_plain_line(f"Select {label}:", as_json=False)
    for index, item in enumerate(items, start=1):
        print_plain_line(
            f"  {index}. {item.get('key')}\t{item.get('name')}",
            as_json=False,
        )
    while True:
        selected = _prompt_choice(f"{label.capitalize()} number")
        if 1 <= selected <= len(items):
            return items[selected - 1]
        print_plain_line("Enter a number from the list.", as_json=False)


def run_jira_use_flow(
    *,
    workspace_key: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Pick a Jira project, persist choice, and fetch issues."""
    if not sys.stdin.isatty() and not workspace_key:
        raise AtlassianReadError(
            "Interactive workspace selection requires a terminal. "
            "Use: potpie jira select --key ENG"
        )
    ctx = load_jira_read_credentials()
    prefs = ctx.get("workspaces") if isinstance(ctx.get("workspaces"), dict) else {}
    items = fetch_jira_projects()
    default_key = str(workspace_key or prefs.get("jira_project") or "").strip().upper()
    if workspace_key or (default_key and (not sys.stdin.isatty() or workspace_key)):
        match = next((i for i in items if i.get("key") == default_key), None)
        picked = match or {"key": default_key, "name": default_key}
    else:
        picked = _prompt_workspace(items, label="Jira project")
    project_key = str(picked.get("key") or "").strip().upper()
    rows = fetch_jira_issues_in_project(project_key, limit=limit)
    save_jira_workspace_prefs(project_key=project_key)
    return {
        "product": "jira",
        "workspace_key": project_key,
        "workspace_name": picked.get("name"),
        "items": rows,
    }


def run_confluence_use_flow(
    *,
    workspace_key: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Pick a Confluence space, persist choice, and fetch pages."""
    if not sys.stdin.isatty() and not workspace_key:
        raise AtlassianReadError(
            "Interactive workspace selection requires a terminal. "
            "Use: potpie confluence select --key DOCS"
        )
    ctx = load_confluence_read_credentials()
    prefs = ctx.get("workspaces") if isinstance(ctx.get("workspaces"), dict) else {}
    items = fetch_confluence_spaces_sample()
    default_key = (
        str(workspace_key or prefs.get("confluence_space") or "").strip().upper()
    )
    if workspace_key or (default_key and (not sys.stdin.isatty() or workspace_key)):
        match = next((i for i in items if i.get("key") == default_key), None)
        picked = match or {"key": default_key, "name": default_key}
    else:
        picked = _prompt_workspace(items, label="Confluence space")
    space_key = str(picked.get("key") or "").strip().upper()
    rows = fetch_confluence_pages_in_space(space_key, limit=limit)
    save_confluence_workspace_prefs(space_key=space_key)
    return {
        "product": "wiki",
        "workspace_key": space_key,
        "workspace_name": picked.get("name"),
        "items": rows,
    }
