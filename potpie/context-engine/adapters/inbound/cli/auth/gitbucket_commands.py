"""Interactive GitBucket personal-access-token login command.

The HTTP client (verification + URL normalization) lives in
``adapters.outbound.cli_auth.gitbucket_client``; this inbound module owns the
interactive flow (prompts, output, exit codes) and mounts the ``gitbucket`` Typer
sub-application.
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser
from collections.abc import Callable
from typing import TypeVar

import click
import typer

from adapters.outbound.cli_auth.credentials_store import (
    ProviderCredentialError,
    credentials_path,
    get_integration_status,
    integration_token_storage,
)
from adapters.outbound.cli_auth.gitbucket_client import (
    GitBucketClientError,
    gitbucket_token_page_url,
    normalize_gitbucket_host_url,
    verify_gitbucket_token,
)
from adapters.outbound.cli_auth.gitbucket_read_client import (
    GitBucketReadError,
    list_gitbucket_repos,
)
from adapters.inbound.cli.commands._common import EXIT_AUTH, EXIT_UNAVAILABLE, get_store
from adapters.inbound.cli.ui.output import emit_error, print_json_blob, print_plain_line
from adapters.outbound.cli_auth.provider_config import (
    GITBUCKET_TOKEN_ENV_VARS,
)
from bootstrap.runtime_settings import ensure_runtime_environment_loaded

T = TypeVar("T")

gitbucket_app = typer.Typer(help="GitBucket integration.")


def _flags() -> tuple[bool, bool]:
    from adapters.inbound.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


def _guard_typer_prompt(callback: Callable[[], T]) -> T:
    """Map Click/Typer Ctrl+C aborts to ``KeyboardInterrupt`` for callers."""
    try:
        return callback()
    except click.Abort:
        raise KeyboardInterrupt from None


def _prompt_host_url() -> str:
    return _guard_typer_prompt(
        lambda: typer.prompt(
            "GitBucket host URL (e.g. https://git.company.com or "
            "https://git.company.com/gitbucket)"
        ).strip()
    )


def _prompt_gitbucket_login() -> str:
    login = _guard_typer_prompt(
        lambda: typer.prompt(
            "GitBucket username (for opening the token page)"
        ).strip()
    )
    if not login:
        raise typer.Exit(code=1)
    return login


def _prompt_token() -> str:
    token = _guard_typer_prompt(
        lambda: typer.prompt("Personal access token", hide_input=True).strip()
    )
    if not token:
        raise typer.Exit(code=1)
    return token


def _open_computed_token_page(token_page: str, *, as_json: bool) -> bool:
    if not as_json:
        print_plain_line(f"Opening {token_page} ...", as_json=False)
    opened = webbrowser.open(token_page, new=1)
    if as_json:
        print_json_blob(
            {
                "ok": True,
                "provider": "gitbucket",
                "action": "open_token_page",
                "token_page_url": token_page,
                "browser_opened": opened,
            },
            as_json=True,
        )
        return opened
    if not opened:
        print_plain_line("Could not open a browser. Open this URL:", as_json=False)
        print_plain_line(token_page, as_json=False, markup=False)
    return opened


def _open_token_page(host_url: str, *, login: str | None = None) -> None:
    """Show setup steps, then open the GitBucket token creation page."""
    print_plain_line("GitBucket login — personal access token", as_json=False)
    for line in (
        "  • Create a token at Account Settings → Applications",
        "  • Press Enter to open the page, then paste your token",
    ):
        print_plain_line(line, as_json=False)
    confirmed = _guard_typer_prompt(
        lambda: typer.confirm(
            "Press Enter to continue",
            default=True,
            show_default=False,
        )
    )
    if not confirmed:
        return

    login_value = (login or "").strip() or _prompt_gitbucket_login()
    token_page = gitbucket_token_page_url(host_url, login_value)
    opened = _open_computed_token_page(token_page, as_json=False)
    if not opened:
        return
    print_plain_line("Paste the token below when you are ready.", as_json=False)


def _token_from_environment() -> str | None:
    for name in GITBUCKET_TOKEN_ENV_VARS:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _token_from_stdin() -> str | None:
    if sys.stdin.isatty():
        return None
    piped = sys.stdin.read()
    if piped and piped.strip():
        return piped.strip()
    return None


def _resolve_gitbucket_token(
    *,
    explicit: str | None = None,
    piped_token: str | None = None,
) -> str | None:
    """Resolve a PAT from env/stdin; ``explicit`` is for programmatic callers/tests only."""
    if (explicit or "").strip():
        return explicit.strip()
    env_token = _token_from_environment()
    if env_token:
        return env_token
    if piped_token is not None:
        return piped_token.strip() or None
    return _token_from_stdin()


def _host_supplied(host: str | None) -> bool:
    return bool((host or "").strip())


def _non_interactive_credentials_available(
    host: str | None,
    token: str | None = None,
) -> bool:
    if not (host or "").strip():
        return False
    if (token or "").strip():
        return True
    if _token_from_environment():
        return True
    return not sys.stdin.isatty()


def _cli_credentials_supplied(
    host: str | None,
    token: str | None,
) -> bool:
    return _non_interactive_credentials_available(host, token)


def run_gitbucket_api_token_auth(
    *,
    force: bool = False,
    as_json: bool = False,
    verbose: bool = False,
    host: str | None = None,
    token: str | None = None,
) -> None:
    """Authenticate with a GitBucket instance using a personal access token."""
    store = get_store()

    try:
        existing = store.get_gitbucket_credentials()
    except ProviderCredentialError as exc:
        emit_error("GitBucket credential lookup failed", str(exc), verbose=verbose)
        existing = {}

    if existing.get("token") and existing.get("host_url") and not force:
        host_url_stored = existing.get("host_url", "")
        account = existing.get("account") if isinstance(existing.get("account"), dict) else {}
        login = str(existing.get("login") or account.get("login") or "").strip()
        print_plain_line(
            "GitBucket is already connected.",
            as_json=as_json,
            json_payload={
                "ok": True,
                "already_connected": True,
                "provider": "gitbucket",
                "host_url": host_url_stored,
                "login": login,
            },
        )
        return

    supplied = _cli_credentials_supplied(host, token)
    if not sys.stdin.isatty() and not supplied:
        emit_error(
            "GitBucket authentication requires credentials",
            "Set GITBUCKET_TOKEN (or POTPIE_GITBUCKET_TOKEN), pipe the token on "
            "stdin with --host, or run in an interactive shell.",
            verbose=verbose,
        )
        raise typer.Exit(code=1)

    if supplied:
        host_value = normalize_gitbucket_host_url((host or "").strip())
        piped_token: str | None = None
        if (
            not (token or "").strip()
            and not _token_from_environment()
            and not sys.stdin.isatty()
        ):
            piped_token = _token_from_stdin()
        token_value = _resolve_gitbucket_token(
            explicit=token,
            piped_token=piped_token,
        )
        if not token_value:
            emit_error(
                "GitBucket authentication failed",
                "Personal access token is required.",
                verbose=verbose,
            )
            raise typer.Exit(code=EXIT_AUTH)
    elif _host_supplied(host):
        host_value = normalize_gitbucket_host_url((host or "").strip())
        if not host_value:
            emit_error(
                "GitBucket authentication failed",
                "Host URL must not be empty.",
                verbose=verbose,
            )
            raise typer.Exit(code=EXIT_AUTH)
        if not as_json:
            _open_token_page(host_value)
        else:
            login_value = _prompt_gitbucket_login()
            _open_computed_token_page(
                gitbucket_token_page_url(host_value, login_value),
                as_json=True,
            )
        token_value = _prompt_token()
    else:
        if not as_json:
            host_value_raw = _prompt_host_url()
            host_value = normalize_gitbucket_host_url(host_value_raw)
            if not host_value:
                emit_error(
                    "GitBucket authentication failed",
                    "Host URL must not be empty.",
                    verbose=verbose,
                )
                raise typer.Exit(code=EXIT_AUTH)
            _open_token_page(host_value)
        else:
            host_value_raw = _prompt_host_url()
            host_value = normalize_gitbucket_host_url(host_value_raw)
            if not host_value:
                emit_error(
                    "GitBucket authentication failed",
                    "Host URL must not be empty.",
                    verbose=verbose,
                )
                raise typer.Exit(code=EXIT_AUTH)
            login_value = _prompt_gitbucket_login()
            token_page = gitbucket_token_page_url(host_value, login_value)
            _open_computed_token_page(token_page, as_json=True)
        token_value = _prompt_token()

    try:
        account_info = verify_gitbucket_token(host_value, token_value)
    except GitBucketClientError as exc:
        emit_error(
            "GitBucket authentication failed",
            str(exc),
            verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH) from exc

    payload = {
        "host_url": host_value,
        "login": account_info.login,
        "email": account_info.email,
        "token": token_value,
        "stored_at": time.time(),
    }
    try:
        store.save_gitbucket_credentials(payload)
    except ProviderCredentialError as exc:
        emit_error("GitBucket credential storage failed", str(exc), verbose=verbose)
        raise typer.Exit(code=EXIT_AUTH) from exc

    token_storage = integration_token_storage()
    stored_status = get_integration_status("gitbucket")
    if stored_status.get("token_storage"):
        token_storage = str(stored_status["token_storage"])
    storage_label = "local credentials file"
    login_display = account_info.login
    email_suffix = f" ({account_info.email})" if account_info.email else ""
    summary = (
        f"Logged in to GitBucket as {login_display}{email_suffix} at {host_value}. "
        f"Stored token in {storage_label}; metadata saved to {credentials_path()}."
    )
    print_plain_line(
        summary,
        as_json=as_json,
        json_payload={
            "ok": True,
            "provider": "gitbucket",
            "host_url": host_value,
            "login": account_info.login,
            "email": account_info.email,
            "path": str(credentials_path()),
            "token_storage": token_storage,
        },
    )


@gitbucket_app.command("login")
def gitbucket_login(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-authenticate even if GitBucket is already connected.",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help=(
            "GitBucket host URL for non-interactive login. Provide the token via "
            "GITBUCKET_TOKEN (or POTPIE_GITBUCKET_TOKEN) or pipe it on stdin."
        ),
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="GitBucket personal access token for non-interactive login.",
    ),
) -> None:
    """Connect to a GitBucket instance with a personal access token."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    run_gitbucket_api_token_auth(
        force=force,
        as_json=j,
        verbose=v,
        host=host,
        token=token,
    )


@gitbucket_app.command("logout")
def gitbucket_logout() -> None:
    """Remove stored GitBucket credentials."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    store = get_store()
    existing = get_integration_status("gitbucket")
    was_authenticated = bool(existing.get("authenticated"))
    try:
        store.clear_gitbucket_credentials()
    except ProviderCredentialError as exc:
        emit_error("GitBucket logout failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc

    if j:
        payload: dict[str, object] = {"ok": True, "provider": "gitbucket"}
        if not was_authenticated:
            payload["cleared_stale"] = True
        print_json_blob(payload, as_json=True)
        return

    if was_authenticated:
        print_plain_line("Logged out successfully.", as_json=False)
        return
    print_plain_line(
        "No active session found. Any stale local credentials were removed.",
        as_json=False,
    )


@gitbucket_app.command("repos")
def gitbucket_repos(
    limit: int = typer.Option(30, "--limit", "-n", min=1, max=100),
) -> None:
    """List GitBucket repositories you can access."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    try:
        repos = list_gitbucket_repos(limit=limit)
    except GitBucketReadError as exc:
        emit_error("GitBucket repository listing failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc

    if j:
        print_json_blob(
            {"ok": True, "provider": "gitbucket", "count": len(repos), "repos": repos},
            as_json=True,
        )
        return

    for repo in repos:
        visibility = "private" if repo.get("private") else "public"
        print_plain_line(
            f"{repo.get('full_name')}\t{visibility}\t{repo.get('default_branch') or ''}",
            as_json=False,
        )


def _build_auth_compat_gitbucket() -> typer.Typer:
    app = typer.Typer(help="[Deprecated] use `potpie gitbucket`.")
    app.command("login")(gitbucket_login)
    app.command("logout")(gitbucket_logout)
    app.command("repos")(gitbucket_repos)
    return app
