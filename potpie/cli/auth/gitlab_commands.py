"""GitLab CLI authentication and read commands."""

from __future__ import annotations

from typing import Any

import typer

from bootstrap.runtime_settings import ensure_runtime_environment_loaded
from potpie.cli.auth.gitlab_auth import run_gitlab_pat_auth
from potpie.cli.auth.gitlab_read import (
    run_gitlab_select_flow,
)
from adapters.outbound.cli_auth.gitlab_read_client import (
    GitLabReadError,
    fetch_gitlab_projects,
)
from potpie.cli.commands._common import EXIT_AUTH, EXIT_UNAVAILABLE
from adapters.outbound.cli_auth.credentials_store import (
    CredentialStoreError,
    ProviderCredentialError,
    get_integration_status,
    list_gitlab_instances,
    clear_gitlab_credentials,
)
from potpie.cli.telemetry.usage_events import (
    capture_usage_command_succeeded,
)
from potpie.cli.ui.output import emit_error, print_json_blob, print_plain_line
from rich.markup import escape

gitlab_app = typer.Typer(help="GitLab integration.")


def _flags() -> tuple[bool, bool]:
    from potpie.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


def _esc(value: Any) -> str:
    return escape(str(value or ""))


@gitlab_app.command("login")
def gitlab_login(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-authenticate even if GitLab is already connected.",
    ),
    instance: str | None = typer.Option(
        None,
        "--instance",
        "-i",
        help="GitLab instance URL (e.g. https://gitlab.corp.com).",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="Personal access token (non-interactive login).",
    ),
) -> None:
    """Connect to a GitLab instance with a Personal Access Token."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    run_gitlab_pat_auth(
        force=force,
        as_json=j,
        verbose=v,
        instance=instance,
        token=token,
    )


def gitlab_logout_impl(instance: str | None = None) -> None:
    """Remove stored GitLab credentials."""
    j, v = _flags()
    was_authenticated = bool(get_integration_status("gitlab").get("authenticated"))
    try:
        clear_gitlab_credentials(instance_host=instance)
    except (ProviderCredentialError, CredentialStoreError, ValueError) as exc:
        emit_error("GitLab logout failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc

    if j:
        payload: dict[str, object] = {"ok": True, "provider": "gitlab"}
        if instance:
            payload["instance"] = instance
        if not was_authenticated:
            payload["cleared_stale"] = True
        print_json_blob(payload, as_json=True)
        return

    if was_authenticated:
        print_plain_line("Logged out successfully.", as_json=False)
    else:
        print_plain_line(
            "No active session found. Any stale local credentials were removed.",
            as_json=False,
        )


@gitlab_app.command("logout")
def gitlab_logout(
    instance: str | None = typer.Option(
        None,
        "--instance",
        "-i",
        help="GitLab instance host to disconnect (default: all).",
    ),
) -> None:
    """Remove stored GitLab credentials."""
    gitlab_logout_impl(instance=instance)


@gitlab_app.command("ls")
def gitlab_ls() -> None:
    """List connected GitLab instances."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    try:
        rows = list_gitlab_instances()
    except (ProviderCredentialError, CredentialStoreError) as exc:
        emit_error("GitLab instance list failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    capture_usage_command_succeeded(
        command="gitlab ls",
        result_kind="provider_list",
        item_count=len(rows),
        provider="gitlab",
    )
    if j:
        print_json_blob(
            {"ok": True, "provider": "gitlab", "instances": rows},
            as_json=True,
        )
        return
    print_plain_line("GitLab instances:", as_json=False)
    if not rows:
        print_plain_line("  (none — run: potpie gitlab login)", as_json=False)
        return
    for row in rows:
        active_suffix = "  (active)" if row.get("active") else ""
        account = row.get("account") or {}
        username = account.get("username") or ""
        host = _esc(row.get("instance_host"))
        user_str = f"  ({_esc(username)})" if username else ""
        print_plain_line(
            f"  {host}{user_str}{active_suffix}",
            as_json=False,
        )
    print_plain_line(
        "\nConnect another instance: potpie gitlab login --instance <url>",
        as_json=False,
    )


@gitlab_app.command("repos")
def gitlab_repos(
    instance: str | None = typer.Option(
        None,
        "--instance",
        "-i",
        help="GitLab instance host.",
    ),
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=100),
) -> None:
    """List GitLab projects accessible to the authenticated user."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    try:
        repos = fetch_gitlab_projects(instance_host=instance, limit=limit)
    except GitLabReadError as exc:
        emit_error("GitLab project listing failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    capture_usage_command_succeeded(
        command="gitlab repos",
        result_kind="provider_list",
        item_count=len(repos),
        provider="gitlab",
    )
    if j:
        print_json_blob(
            {"ok": True, "provider": "gitlab", "count": len(repos), "projects": repos},
            as_json=True,
        )
        return
    print_plain_line(f"GitLab projects ({len(repos)}):", as_json=False)
    if not repos:
        print_plain_line("  (none)", as_json=False)
        return
    for repo in repos:
        vis = repo.get("visibility") or ""
        vis_label = f" [{vis}]" if vis else ""
        print_plain_line(
            f"  {_esc(repo.get('path_with_namespace'))}{vis_label}",
            as_json=False,
        )


@gitlab_app.command("select")
def gitlab_select(
    instance: str | None = typer.Option(
        None,
        "--instance",
        "-i",
        help="GitLab instance host.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        "-p",
        help="GitLab project path (e.g. group/repo).",
    ),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
) -> None:
    """Select a GitLab project, then fetch MRs and issues."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    try:
        result = run_gitlab_select_flow(
            instance_host=instance,
            project_path=project,
            limit=limit,
        )
    except GitLabReadError as exc:
        emit_error("GitLab fetch failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc

    capture_usage_command_succeeded(
        command="gitlab select",
        result_kind="provider_select",
        item_count=(
            len(result.get("merge_requests") or []) + len(result.get("issues") or [])
        ),
        provider="gitlab",
    )

    if j:
        print_json_blob({"ok": True, "provider": "gitlab", **result}, as_json=True)
        return

    project_label = result.get("workspace_key") or result.get("workspace_name")
    print_plain_line(
        f"Project: {_esc(project_label)}",
        as_json=False,
    )

    mrs = result.get("merge_requests") or []
    if mrs:
        print_plain_line(f"\nOpen merge requests ({len(mrs)}):", as_json=False)
        for mr in mrs:
            _print_mr_row(mr)
    else:
        print_plain_line("\nNo open merge requests.", as_json=False)

    issues = result.get("issues") or []
    if issues:
        print_plain_line(f"\nOpen issues ({len(issues)}):", as_json=False)
        for issue in issues:
            _print_issue_row(issue)
    else:
        print_plain_line("\nNo open issues.", as_json=False)


def _print_mr_row(row: dict[str, Any]) -> None:
    iid = _esc(row.get("iid"))
    title = _esc(row.get("title"))
    print_plain_line(f"\n  !{iid}  {title}".rstrip(), as_json=False)
    if row.get("author"):
        print_plain_line(f"    Author: {_esc(row.get('author'))}", as_json=False)
    if row.get("source_branch"):
        print_plain_line(
            f"    {_esc(row.get('source_branch'))} → {_esc(row.get('target_branch'))}",
            as_json=False,
        )
    if row.get("web_url"):
        print_plain_line(f"    URL: {_esc(row.get('web_url'))}", as_json=False)


def _print_issue_row(row: dict[str, Any]) -> None:
    iid = _esc(row.get("iid"))
    title = _esc(row.get("title"))
    print_plain_line(f"\n  #{iid}  {title}".rstrip(), as_json=False)
    if row.get("assignee"):
        print_plain_line(f"    Assignee: {_esc(row.get('assignee'))}", as_json=False)
    if row.get("labels"):
        labels = row.get("labels") or []
        if isinstance(labels, list) and labels:
            print_plain_line(
                f"    Labels: {', '.join(_esc(label) for label in labels)}",
                as_json=False,
            )
    if row.get("web_url"):
        print_plain_line(f"    URL: {_esc(row.get('web_url'))}", as_json=False)
