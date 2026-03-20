"""potpie parse – submit a repository for parsing and poll until complete."""

from __future__ import annotations

import sys
from pathlib import Path

from potpie.cli.client import PotpieClient, PotpieClientError


def parse_repo(
    repo_path: str,
    branch: str = "main",
    base_url: str | None = None,
) -> None:
    """Parse a local repository and display live status updates.

    Args:
        repo_path: Absolute or relative path to the repository on disk.
        branch: Branch name to parse (default: ``"main"``).
        base_url: Override the server URL (default: ``http://localhost:8001``).
    """
    # Validate repo path
    resolved = Path(repo_path).expanduser().resolve()
    if not resolved.exists():
        print(f"Error: repository path does not exist: {resolved}", file=sys.stderr)
        sys.exit(1)
    if not resolved.is_dir():
        print(f"Error: path is not a directory: {resolved}", file=sys.stderr)
        sys.exit(1)

    client = PotpieClient(base_url) if base_url else PotpieClient()

    print(f"Submitting repository for parsing: {resolved}")
    print(f"Branch: {branch}")

    try:
        result = client.parse(repo_path=str(resolved), branch_name=branch)
    except PotpieClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    project_id = result.get("project_id")
    status = result.get("status", "unknown")

    if not project_id:
        print(f"Error: server did not return a project_id: {result}", file=sys.stderr)
        sys.exit(1)

    print(f"Project ID : {project_id}")
    print(f"Status     : {status}")

    if status.lower() == "ready":
        print("Repository is already parsed and ready.")
        return

    print("\nPolling for parsing status (Ctrl+C to stop polling)…")
    try:
        for status_data in client.poll_parsing_status(project_id):
            current = (status_data.get("status") or "unknown").lower()
            print(f"  status: {current}")
            if current == "ready":
                print(f"\nParsing complete. Project ID: {project_id}")
                return
            if current in ("error", "failed"):
                print(f"\nParsing failed with status: {current}", file=sys.stderr)
                sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nStopped polling. Project ID: {project_id}")
    except PotpieClientError as exc:
        print(f"\nError while polling: {exc}", file=sys.stderr)
        sys.exit(1)
