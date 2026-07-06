"""Interactive GitLab read flows (project picker and MR/issue fetch).

The HTTP/data layer lives in ``adapters.outbound.cli_auth.gitlab_read_client``;
this inbound module owns the interactive project-selection flow.
"""

from __future__ import annotations

import sys
from typing import Any

import typer

from adapters.outbound.cli_auth.gitlab_read_client import (
    GitLabReadError,
    fetch_gitlab_issues,
    fetch_gitlab_merge_requests,
    fetch_gitlab_projects,
    load_gitlab_read_credentials,
)
from adapters.outbound.cli_auth.credentials_store import (
    save_gitlab_workspace_prefs,
)
from adapters.inbound.cli.ui.output import print_plain_line


def _prompt_choice(label: str, *, default: str = "1") -> int:
    while True:
        choice = typer.prompt(label, default=default).strip()
        try:
            return int(choice)
        except ValueError:
            print_plain_line("Enter a number from the list.", as_json=False)


def _prompt_project(
    projects: list[dict[str, Any]],
) -> dict[str, Any]:
    if not projects:
        raise GitLabReadError("No GitLab projects found for this account.")
    print_plain_line("Select a project:", as_json=False)
    for index, p in enumerate(projects, start=1):
        print_plain_line(
            f"  {index}. {p.get('path_with_namespace')}\t{p.get('name')}",
            as_json=False,
        )
    while True:
        selected = _prompt_choice("Project number")
        if 1 <= selected <= len(projects):
            return projects[selected - 1]
        print_plain_line("Enter a number from the list.", as_json=False)


def run_gitlab_select_flow(
    *,
    instance_host: str | None = None,
    project_path: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Pick a GitLab project, persist choice, and fetch sample MRs + issues."""
    creds = load_gitlab_read_credentials(instance_host=instance_host)
    prefs = creds.get("workspaces") if isinstance(creds.get("workspaces"), dict) else {}
    default_project = str(project_path or prefs.get("default_project") or "").strip()

    if not sys.stdin.isatty() and not project_path and not default_project:
        raise GitLabReadError(
            "Interactive project selection requires a terminal. "
            "Use: potpie gitlab select --project group/repo"
        )
    projects = fetch_gitlab_projects(instance_host=instance_host, limit=100)

    if project_path or (default_project and not sys.stdin.isatty()):
        match = next(
            (p for p in projects if p.get("path_with_namespace") == default_project),
            None,
        )
        picked = match or {
            "path_with_namespace": default_project,
            "name": default_project,
            "id": None,
        }
    else:
        picked = _prompt_project(projects)

    path_with_ns = str(picked.get("path_with_namespace") or "").strip()
    project_id = picked.get("id")

    if project_id is None:
        project_id = path_with_ns

    mrs = fetch_gitlab_merge_requests(
        project_id,
        instance_host=instance_host,
        limit=limit,
    )
    issues = fetch_gitlab_issues(
        project_id,
        instance_host=instance_host,
        limit=limit,
    )

    save_gitlab_workspace_prefs(
        instance_host=instance_host,
        default_project=path_with_ns,
    )
    return {
        "product": "gitlab",
        "workspace_key": path_with_ns,
        "workspace_name": picked.get("name"),
        "merge_requests": mrs,
        "issues": issues,
    }
