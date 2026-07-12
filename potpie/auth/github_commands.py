"""GitHub CLI authentication commands (device flow)."""

from __future__ import annotations

import select
import sys
import time
import webbrowser
from typing import NoReturn

import typer
from click.exceptions import Abort

from potpie.auth.github import (
    GitHubDeviceFlowError,
    build_provider_credentials,
    list_user_owned_repositories,
    poll_for_access_token,
    request_device_code,
    verify_account,
)
from potpie.runtime.settings import ensure_runtime_environment_loaded
from potpie.cli.commands._common import EXIT_AUTH, EXIT_UNAVAILABLE, get_store
from potpie.auth.credentials_store import (
    CredentialStoreError,
    ProviderCredentialError,
    _integration_secret_store_label,
    get_integration_status,
)
from potpie.cli.telemetry.onboarding_events import (
    capture_github_auth_event,
    current_entrypoint,
    elapsed_ms,
    now_ms,
    sanitized_failure_kind,
)
from potpie.cli.telemetry.usage_events import (
    capture_usage_command_succeeded,
)
from potpie.cli.ui.output import emit_error, print_json_blob, print_plain_line

if sys.platform == "win32":
    import msvcrt
else:
    msvcrt = None

EXIT_CANCELLED = 130
GITHUB_AUTO_OPEN_SECONDS = 10
_GITHUB_OPEN_PROMPT_PREFIX = (
    "Copy the code. Press Enter to open now, or GitHub opens in "
)

github_app = typer.Typer(help="GitHub integration.")


def _flags() -> tuple[bool, bool]:
    from potpie.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


def _open_github_device_verification(user_code: str, verification_uri: str) -> bool:
    print_plain_line(
        "GitHub login requires a one-time verification code.",
        as_json=False,
    )
    print_plain_line(f"Copy this code: {user_code}", as_json=False)
    try:
        _wait_for_enter_or_auto_open()
    except BaseException as exc:
        if _is_github_login_cancel(exc):
            _cancel_github_login()
        raise
    print_plain_line("Opening GitHub now...", as_json=False)

    opened = webbrowser.open(verification_uri)
    if not opened:
        print_plain_line(
            "Could not open a browser automatically. Open this URL:",
            as_json=False,
        )
        print_plain_line(verification_uri, as_json=False, markup=False)
        return False
    print_plain_line(
        "Paste the copied code into GitHub to continue.",
        as_json=False,
    )
    return True


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
    seconds: int = GITHUB_AUTO_OPEN_SECONDS,
    input_stream=None,
) -> None:
    input_stream = input_stream or sys.stdin
    use_windows_stdin = input_stream is sys.stdin and sys.platform == "win32"
    for remaining in range(seconds, 0, -1):
        _write_inline_prompt(f"{_GITHUB_OPEN_PROMPT_PREFIX}{remaining}s")
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


def github_login_impl() -> None:
    """Authenticate the CLI with GitHub using device flow."""
    ensure_runtime_environment_loaded()
    j, v = _flags()
    entrypoint = current_entrypoint("direct_github_auth")
    auth_started_ms = now_ms()
    stage = "started"
    capture_github_auth_event(
        "cli_onboarding_github_auth_started",
        entrypoint=entrypoint,
    )
    account = None
    payload = None
    try:
        store = get_store()
        stage = "device_code"
        stage_started_ms = now_ms()
        device_code = request_device_code()
        capture_github_auth_event(
            "cli_onboarding_github_device_code_requested",
            entrypoint=entrypoint,
            duration_ms=elapsed_ms(stage_started_ms),
        )
        if not j:
            stage_started_ms = now_ms()
            opened = _open_github_device_verification(
                device_code.user_code,
                device_code.verification_uri,
            )
            capture_github_auth_event(
                "cli_onboarding_github_browser_open_attempted",
                entrypoint=entrypoint,
                duration_ms=elapsed_ms(stage_started_ms),
                browser_opened=opened,
            )
            print_plain_line("Waiting for authorization...", as_json=False)
        stage = "token_poll"
        capture_github_auth_event(
            "cli_onboarding_github_token_poll_started",
            entrypoint=entrypoint,
        )
        stage_started_ms = now_ms()
        token = poll_for_access_token(device_code)
        capture_github_auth_event(
            "cli_onboarding_github_token_poll_completed",
            entrypoint=entrypoint,
            duration_ms=elapsed_ms(stage_started_ms),
        )
        stage = "account_verified"
        stage_started_ms = now_ms()
        account = verify_account(token.access_token)
        capture_github_auth_event(
            "cli_onboarding_github_account_verified",
            entrypoint=entrypoint,
            duration_ms=elapsed_ms(stage_started_ms),
        )
        payload = build_provider_credentials(
            token, account, verification_uri=device_code.verification_uri
        )
        stage = "credentials_stored"
        stage_started_ms = now_ms()
        store.write_provider_credentials("github", payload.as_dict())
        capture_github_auth_event(
            "cli_onboarding_github_credentials_stored",
            entrypoint=entrypoint,
            duration_ms=elapsed_ms(stage_started_ms),
        )
    except GitHubDeviceFlowError as exc:
        capture_github_auth_event(
            "cli_onboarding_github_auth_failed",
            entrypoint=entrypoint,
            duration_ms=elapsed_ms(auth_started_ms),
            failure_stage=stage,
            failure_kind=sanitized_failure_kind(exc),
        )
        emit_error("GitHub login failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except (ProviderCredentialError, CredentialStoreError) as exc:
        capture_github_auth_event(
            "cli_onboarding_github_auth_failed",
            entrypoint=entrypoint,
            duration_ms=elapsed_ms(auth_started_ms),
            failure_stage=stage,
            failure_kind=sanitized_failure_kind(exc),
        )
        emit_error("GitHub login failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except typer.Exit:
        raise
    except (Abort, KeyboardInterrupt, EOFError) as exc:
        if _is_github_login_cancel(exc):
            _cancel_github_login()
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_github_login_cancel(exc):
            _cancel_github_login()
        capture_github_auth_event(
            "cli_onboarding_github_auth_failed",
            entrypoint=entrypoint,
            duration_ms=elapsed_ms(auth_started_ms),
            failure_stage=stage,
            failure_kind=sanitized_failure_kind(exc),
        )
        _capture_unexpected_auth_error(exc, title="GitHub login failed", verbose=v)
    capture_github_auth_event(
        "cli_onboarding_github_auth_completed",
        entrypoint=entrypoint,
        duration_ms=elapsed_ms(auth_started_ms),
    )

    if j:
        print_json_blob(
            {
                "ok": True,
                "provider": "github",
                "account": payload.account if payload else {},
                "path": str(store.credentials_path()),
            },
            as_json=True,
        )
        return

    if account is None or payload is None:
        _capture_unexpected_auth_error(
            RuntimeError("GitHub login completed without account credentials."),
            title="GitHub login failed",
            verbose=v,
        )
    stored = store.get_provider_credentials("github")
    login = str(stored.get("account", {}).get("login") or account.login)
    email = str(stored.get("account", {}).get("email") or account.email or "")
    print_plain_line(
        f"Logged in to GitHub as {login}" + (f" ({email})" if email else ""),
        as_json=False,
    )


def _cancel_github_login() -> NoReturn:
    typer.echo()
    print_plain_line("GitHub login cancelled.", as_json=False)
    raise typer.Exit(code=EXIT_CANCELLED) from None


def _is_github_login_cancel(exc: BaseException) -> bool:
    return isinstance(exc, (Abort, KeyboardInterrupt, EOFError)) or (
        exc.__class__.__name__ == "Abort"
    )


def github_logout_impl() -> None:
    """Remove GitHub credentials from local credential files."""
    j, v = _flags()
    store = get_store()
    was_authenticated = bool(get_integration_status("github").get("authenticated"))
    try:
        store = get_store()
        store.clear_provider_credentials("github")
    except (ProviderCredentialError, CredentialStoreError, ValueError) as exc:
        emit_error("GitHub logout failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _capture_unexpected_auth_error(exc, title="GitHub logout failed", verbose=v)

    if j:
        payload: dict[str, object] = {"ok": True, "provider": "github"}
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


@github_app.command("login")
def github_login_cmd() -> None:
    """Authenticate with GitHub using device flow."""
    github_login_impl()


@github_app.command("logout")
def github_logout_cmd() -> None:
    """Remove stored GitHub credentials."""
    github_logout_impl()


def github_list_impl() -> None:
    """List GitHub repositories owned by the authenticated user."""
    j, v = _flags()
    try:
        store = get_store()
        credentials = store.get_provider_credentials("github")
    except (ProviderCredentialError, CredentialStoreError) as exc:
        emit_error(
            "GitHub credentials not found",
            str(exc),
            verbose=v,
        )
        raise typer.Exit(code=EXIT_AUTH) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _capture_unexpected_auth_error(
            exc,
            title="GitHub test repos failed",
            verbose=v,
        )
    token = str(credentials.get("access_token") or "").strip()
    if not token:
        emit_error(
            "GitHub credentials not found",
            f"GitHub token not found in {_integration_secret_store_label()}. "
            "Run: potpie integration github login",
            verbose=v,
        )
        raise typer.Exit(code=EXIT_AUTH)
    try:
        repos = list_user_owned_repositories(token)
    except GitHubDeviceFlowError as exc:
        emit_error("GitHub repository listing failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _capture_unexpected_auth_error(
            exc,
            title="GitHub repository listing failed",
            verbose=v,
        )
    capture_usage_command_succeeded(
        command="integration.github.list",
        result_kind="provider_list",
        item_count=len(repos),
        provider="github",
    )

    if j:
        print_json_blob(
            {"ok": True, "provider": "github", "count": len(repos), "repos": repos},
            as_json=True,
        )
        return

    for repo in repos:
        visibility = "private" if repo.get("private") else "public"
        print_plain_line(
            f"{repo.get('full_name')}\t{visibility}\t{repo.get('default_branch') or ''}",
            as_json=False,
        )


def _capture_unexpected_auth_error(
    exc: BaseException,
    *,
    title: str,
    verbose: bool,
) -> NoReturn:
    from potpie.cli.telemetry.sentry_runtime import (
        capture_unexpected_cli_error,
    )

    capture_unexpected_cli_error(
        exc,
        error_code="unexpected_cli_error",
        error_kind="unexpected",
    )
    emit_error(
        title,
        "Unexpected internal error.",
        code="unexpected_cli_error",
        verbose=verbose,
    )
    raise typer.Exit(code=EXIT_AUTH) from exc


@github_app.command("list")
def github_list_cmd() -> None:
    """List GitHub repositories you can access."""
    github_list_impl()
