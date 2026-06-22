"""Interactive Bitbucket read flows (workspace + repository pickers)."""

from __future__ import annotations

import sys
from typing import Any

from adapters.inbound.cli.auth.atlassian_read import _prompt_workspace
from adapters.outbound.cli_auth.bitbucket_read_client import (
    BitbucketReadError,
    fetch_bitbucket_pull_requests,
    fetch_bitbucket_repositories,
    fetch_bitbucket_workspaces,
    load_bitbucket_read_credentials,
)
from adapters.outbound.cli_auth.credentials_store import save_bitbucket_workspace_prefs


def run_bitbucket_use_flow(
    *,
    workspace_key: str | None = None,
    repo_slug: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Pick a Bitbucket workspace and repository, persist choice, and fetch PRs."""
    if not sys.stdin.isatty() and not (workspace_key and repo_slug):
        raise BitbucketReadError(
            "Interactive workspace selection requires a terminal. "
            "Use: potpie bitbucket select --workspace potpie --repo backend"
        )
    ctx = load_bitbucket_read_credentials()
    prefs = ctx.get("workspaces") if isinstance(ctx.get("workspaces"), dict) else {}

    workspaces = fetch_bitbucket_workspaces()
    default_workspace = str(
        workspace_key or prefs.get("bitbucket_workspace") or ""
    ).strip()
    if workspace_key or (default_workspace and not sys.stdin.isatty()):
        picked_workspace = next(
            (item for item in workspaces if str(item.get("key") or "").strip() == default_workspace),
            None,
        ) or {"key": default_workspace, "name": default_workspace}
    elif len(workspaces) == 1:
        picked_workspace = workspaces[0]
    else:
        picked_workspace = _prompt_workspace(workspaces, label="Bitbucket workspace")

    workspace_value = str(picked_workspace.get("key") or "").strip()
    if not workspace_value:
        raise BitbucketReadError("Could not resolve a Bitbucket workspace.")

    repositories = fetch_bitbucket_repositories(workspace_value, limit=50)
    default_repo = str(repo_slug or prefs.get("bitbucket_repository") or "").strip()
    if repo_slug or (default_repo and not sys.stdin.isatty()):
        picked_repo = next(
            (item for item in repositories if str(item.get("key") or "").strip() == default_repo),
            None,
        ) or {"key": default_repo, "name": default_repo}
    elif len(repositories) == 1:
        picked_repo = repositories[0]
    else:
        picked_repo = _prompt_workspace(repositories, label="Bitbucket repository")

    repo_value = str(picked_repo.get("key") or "").strip()
    if not repo_value:
        raise BitbucketReadError("Could not resolve a Bitbucket repository.")

    rows = fetch_bitbucket_pull_requests(
        workspace_value,
        repo_value,
        limit=limit,
    )
    save_bitbucket_workspace_prefs(
        workspace_key=workspace_value,
        repo_slug=repo_value,
    )
    return {
        "product": "bitbucket",
        "workspace_key": workspace_value,
        "workspace_name": picked_workspace.get("name"),
        "repo_key": repo_value,
        "repo_name": picked_repo.get("name"),
        "items": rows,
    }
