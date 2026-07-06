"""Interactive GitLab PAT login command.

The HTTP client (verification + profile fetch) lives in
``adapters.outbound.cli_auth.gitlab_client``; this inbound module owns the
interactive flow (prompts, output, exit codes).
"""

from __future__ import annotations

import select
import sys
import time
import webbrowser
from typing import Any, NoReturn

import typer

import click
from click.exceptions import Abort

from adapters.outbound.cli_auth.gitlab_client import (
    GitLabAuthErrorKind,
    instance_host,
    normalize_instance_url,
    parse_user_profile,
    verify_instance_access,
    verify_read_api_scope,
)
from adapters.outbound.cli_auth.credentials_store import (
    ProviderCredentialError,
    credentials_path,
    get_gitlab_credentials,
    get_integration_status,
    save_gitlab_credentials,
)
from adapters.inbound.cli.commands._common import EXIT_AUTH, get_store
from adapters.inbound.cli.ui.output import emit_error, print_plain_line
from adapters.outbound.cli_auth.provider_config import (
    GITLAB_DEFAULT_INSTANCE,
    GITLAB_RECOMMENDED_SCOPES,
    gitlab_pat_page_url,
)

if sys.platform == "win32":
    import msvcrt
else:
    msvcrt = None

EXIT_CANCELLED = 130
GITLAB_AUTO_OPEN_SECONDS = 10
_GITLAB_OPEN_PROMPT_PREFIX = (
    "Press Enter to open now, or browser opens in "
)


def _guard_typer_prompt(callback):
    try:
        return callback()
    except click.Abort:
        raise KeyboardInterrupt from None


def _prompt_instance_url(default: str = "") -> str:
    prompt_default = default or "gitlab.com"
    raw = _guard_typer_prompt(
        lambda: typer.prompt(
            "Enter your GitLab instance URL",
            default=prompt_default,
        ).strip()
    )
    if raw in ("gitlab.com", "https://gitlab.com"):
        return GITLAB_DEFAULT_INSTANCE
    return raw


def _prompt_pat() -> str:
    pat = _guard_typer_prompt(
        lambda: typer.prompt(
            "Enter your personal access token",
            hide_input=True,
        ).strip()
    )
    if not pat:
        raise typer.Exit(code=1)
    return pat


def _detect_gitlab_from_git_remote() -> str | None:
    """Try to guess the GitLab instance from the current repo's git remote."""
    try:
        from adapters.inbound.cli.repo_location import current_git_remote
        from pathlib import Path

        remote = current_git_remote(Path.cwd().resolve())
        if remote and "gitlab" in remote.lower():
            host = remote.split("/")[0] if "/" in remote else ""
            if host and "gitlab" in host.lower():
                return f"https://{host}"
    except Exception:
        pass
    return None


def _write_inline_prompt(message: str) -> None:
    sys.stdout.write(f"\r\033[K{message}")
    sys.stdout.flush()


def _finish_inline_prompt() -> None:
    sys.stdout.write("\r\033[K\n")
    sys.stdout.flush()


def _stdin_enter_pressed_windows(*, timeout: float) -> bool:
    if msvcrt is None:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in ("\r", "\n"):
                return True
        time.sleep(0.05)
    return False


def _wait_for_enter_or_auto_open(
    *,
    seconds: int = GITLAB_AUTO_OPEN_SECONDS,
    input_stream=None,
) -> None:
    input_stream = input_stream or sys.stdin
    use_windows_stdin = input_stream is sys.stdin and sys.platform == "win32"
    for remaining in range(seconds, 0, -1):
        _write_inline_prompt(f"{_GITLAB_OPEN_PROMPT_PREFIX}{remaining}s")
        if use_windows_stdin:
            if _stdin_enter_pressed_windows(timeout=1.0):
                _finish_inline_prompt()
                return
            continue
        try:
            ready, _, _ = select.select([input_stream], [], [], 1)
        except (OSError, TypeError, ValueError):
            time.sleep(1)
            ready = []
        if ready:
            input_stream.readline()
            _finish_inline_prompt()
            return
    _finish_inline_prompt()


def _cancel_gitlab_login() -> NoReturn:
    typer.echo()
    print_plain_line("GitLab login cancelled.", as_json=False)
    raise typer.Exit(code=EXIT_CANCELLED) from None


def _is_gitlab_login_cancel(exc: BaseException) -> bool:
    return isinstance(exc, (Abort, KeyboardInterrupt, EOFError)) or (
        exc.__class__.__name__ == "Abort"
    )


def _auth_failure_message(
    error_kind: GitLabAuthErrorKind | None,
    instance_url: str,
) -> str:
    scopes = ", ".join(GITLAB_RECOMMENDED_SCOPES)
    lines = [
        "Could not authenticate with GitLab.",
        f"  - Instance: {instance_url}",
        f"  - Required scopes: {scopes}",
    ]
    if error_kind == GitLabAuthErrorKind.INVALID_CREDENTIALS:
        lines.insert(1, "  - Invalid personal access token")
    elif error_kind == GitLabAuthErrorKind.INSUFFICIENT_SCOPES:
        lines.insert(1, f"  - Token is missing required scopes ({scopes})")
    elif error_kind == GitLabAuthErrorKind.INSTANCE_UNREACHABLE:
        lines.insert(1, f"  - Cannot reach {instance_url} (check URL, DNS, TLS)")
    return "\n".join(lines)


def _open_gitlab_pat_page(inst_url: str) -> None:
    """Show setup steps, then open the GitLab PAT page after countdown or Enter."""
    scopes = ", ".join(GITLAB_RECOMMENDED_SCOPES)
    print_plain_line("GitLab login — Personal Access Token", as_json=False)
    for line in (
        f"  • Create a token at your GitLab instance with scope: {scopes}",
        "  • One token per instance; it works for projects, MRs, and issues",
        f"  • Press Enter to open now, or the browser opens in "
        f"{GITLAB_AUTO_OPEN_SECONDS}s",
    ):
        print_plain_line(line, as_json=False)
    try:
        _wait_for_enter_or_auto_open()
    except BaseException as exc:
        if _is_gitlab_login_cancel(exc):
            _cancel_gitlab_login()
        raise

    pat_url = gitlab_pat_page_url(inst_url)
    print_plain_line("Opening token settings now...", as_json=False)

    opened = webbrowser.open(pat_url, new=1)
    if not opened:
        print_plain_line(
            "Could not open a browser. Open this URL:",
            as_json=False,
        )
        print_plain_line(pat_url, as_json=False, markup=False)
        return
    print_plain_line("Paste the token below when you are ready.", as_json=False)


def run_gitlab_pat_auth(
    *,
    force: bool = False,
    as_json: bool = False,
    verbose: bool = False,
    instance: str | None = None,
    token: str | None = None,
) -> None:
    """Authenticate with a GitLab instance using a Personal Access Token."""
    supplied = bool((instance or "").strip() and (token or "").strip())

    if not force:
        host_to_check = (
            instance_host(instance) if instance else None
        )
        existing = get_gitlab_credentials(instance_host=host_to_check)
        if existing.get("personal_access_token"):
            inst = existing.get("instance_url") or existing.get("instance_host", "")
            print_plain_line(
                f"GitLab is already connected to {inst}.",
                as_json=as_json,
                json_payload={
                    "ok": True,
                    "already_connected": True,
                    "provider": "gitlab",
                    "instance_url": inst,
                },
            )
            return

    if not sys.stdin.isatty() and not supplied:
        emit_error(
            "GitLab authentication requires a terminal",
            "Run in an interactive shell, or pass --instance and --token.",
            verbose=verbose,
        )
        raise typer.Exit(code=1)

    if supplied:
        inst_url = normalize_instance_url(instance or "") or GITLAB_DEFAULT_INSTANCE
        pat = (token or "").strip()
    else:
        git_hint = _detect_gitlab_from_git_remote()
        inst_url = normalize_instance_url(
            _prompt_instance_url(default=instance or git_hint or "")
        ) or GITLAB_DEFAULT_INSTANCE
        if not as_json:
            _open_gitlab_pat_page(inst_url)
        else:
            webbrowser.open(gitlab_pat_page_url(inst_url), new=1)
        pat = _prompt_pat()

    ok, error_kind, user_data = verify_instance_access(inst_url, pat)
    if not ok:
        emit_error(
            "GitLab authentication failed",
            _auth_failure_message(error_kind, inst_url),
            verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH)

    scope_ok, scope_error = verify_read_api_scope(inst_url, pat)
    if not scope_ok:
        emit_error(
            "GitLab authentication failed",
            _auth_failure_message(scope_error, inst_url),
            verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH)

    account = parse_user_profile(user_data)
    host = instance_host(inst_url)

    payload: dict[str, Any] = {
        "auth_type": "personal_access_token",
        "instance_url": inst_url,
        "instance_host": host,
        "personal_access_token": pat,
        "stored_at": time.time(),
    }

    try:
        save_gitlab_credentials(payload, account=account)
    except ProviderCredentialError as exc:
        emit_error(
            "GitLab credential storage failed", str(exc), verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH) from exc

    username = account.get("username") or account.get("name") or "user"
    summary = (
        f"Connected GitLab ({host}) as {username}. "
        f"Stored token in local credentials; metadata saved to {credentials_path()}."
    )
    print_plain_line(
        summary,
        as_json=as_json,
        json_payload={
            "ok": True,
            "provider": "gitlab",
            "instance_url": inst_url,
            "instance_host": host,
            "username": username,
            "path": str(credentials_path()),
            "token_storage": "file",
        },
    )
