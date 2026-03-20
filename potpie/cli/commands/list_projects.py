"""potpie list-projects – display all projects for the current user."""

from __future__ import annotations

import os
import sys

from potpie.cli.client import PotpieClient, PotpieClientError


def list_projects(base_url: str | None = None) -> None:
    """Fetch and display all projects registered on the local Potpie server.

    Args:
        base_url: Override the server URL (default: ``http://localhost:8001``).
    """
    client = PotpieClient(base_url or os.getenv("POTPIE_BASE_URL", "http://localhost:8001"))

    try:
        projects = client.list_projects()
    except PotpieClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not projects:
        print("No projects found.")
        return

    print(f"{'ID':<40}  {'REPO':<35}  {'BRANCH':<20}  STATUS")
    print("-" * 110)
    for project in projects:
        pid = project.get("id", project.get("project_id", ""))
        repo = project.get("repo_name") or project.get("repo_path") or ""
        branch = project.get("branch_name") or ""
        status = project.get("status") or ""
        print(f"{pid:<40}  {repo:<35}  {branch:<20}  {status}")
