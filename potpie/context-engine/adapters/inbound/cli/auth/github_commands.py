"""GitHub CLI authentication commands (device flow)."""

from __future__ import annotations

import select
import sys
import time
import webbrowser
from typing import NoReturn

from click.exceptions import Abort
import typer

from adapters.outbound.cli_auth.env_bootstrap import load_cli_env
from adapters.outbound.cli_auth.github import (
    GitHubDeviceFlowError,
    build_provider_credentials,
    list_user_owned_repositories,
    poll_for_access_token,
    request_device_code,
    verify_account,
)
from adapters.inbound.cli.commands._common import EXIT_AUTH, EXIT_UNAVAILABLE, get_store
from adapters.outbound.cli_auth.credentials_store import (
    CredentialStoreError,
    ProviderCredentialError,
    _integration_secret_store_label,
    get_integration_status,
)
from adapters.inbound.cli.ui.output import emit_error, print_json_blob, print_plain_line

EXIT_CANCELLED = 130
GITHUB_AUTO_OPEN_SECONDS = 10
_GITHUB_OPEN_PROMPT_PREFIX = (
    "Copy the code. Press Enter to open now, or GitHub opens in "
)

github_app = typer.Typer(help="GitHub integration.")
github_test_app = typer.Typer(
    help="Deprecated: use `potpie github repos`.",
    hidden=True,
)
git_app = typer.Typer(
    help="Deprecated: use `potpie github` instead.",
    hidden=True,
)
git_test_app = typer.Typer(help="Deprecated: use `potpie github repos`.", hidden=True)


def _flags() -> tuple[bool, bool]:
    from adapters.inbound.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


def _open_github_device_verification(user_code: str, verification_uri: str) -> None:
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
        return
    print_plain_line(
        "Paste the copied code into GitHub to continue.",
        as_json=False,
    )


def _write_inline_prompt(message: str) -> None:
    sys.stdout.write(f"\r\033[K{message}")
    sys.stdout.flush()


def _finish_inline_prompt() -> None:
    sys.stdout.write("\r\033[K\n")
    sys.stdout.flush()


def _wait_for_enter_or_auto_open(
    *,
    seconds: int = GITHUB_AUTO_OPEN_SECONDS,
    input_stream=None,
) -> None:
    input_stream = input_stream or sys.stdin
    for remaining in range(seconds, 0, -1):
        _write_inline_prompt(f"{_GITHUB_OPEN_PROMPT_PREFIX}{remaining}s")
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
    load_cli_env()
    j, v = _flags()
    account = None
    payload = None
    try:
        store = get_store()
        device_code = request_device_code()
        if not j:
            _open_github_device_verification(
                device_code.user_code,
                device_code.verification_uri,
            )
            print_plain_line("Waiting for authorization...", as_json=False)
        token = poll_for_access_token(device_code)
        account = verify_account(token.access_token)
        payload = build_provider_credentials(
            token, account, verification_uri=device_code.verification_uri
        )
        store.write_provider_credentials("github", payload.as_dict())
    except GitHubDeviceFlowError as exc:
        emit_error("GitHub login failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except (ProviderCredentialError, CredentialStoreError) as exc:
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
        _capture_unexpected_auth_error(exc, title="GitHub login failed", verbose=v)

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

    assert account is not None and payload is not None
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


def _capture_unexpected_auth_error(
    exc: Exception,
    *,
    title: str,
    verbose: bool,
) -> NoReturn:
    from adapters.inbound.cli.sentry_runtime import capture_unexpected_cli_error

    capture_unexpected_cli_error(
        exc,
        error_code="unexpected_cli_error",
        error_kind="unexpected",
    )
    emit_error(
        title,
        "Unexpected error. Run with --verbose for details.",
        verbose=verbose,
        exc=exc,
    )
    raise typer.Exit(code=EXIT_AUTH) from exc


def github_logout_impl() -> None:
    """Remove GitHub credentials from keychain and config."""
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

    from adapters.inbound.cli.auth.auth_commands import _print_standard_logout

    _print_standard_logout(
        was_authenticated=was_authenticated,
        provider="github",
        j=False,
    )


@github_app.command("login")
def github_login_cmd() -> None:
    """Authenticate with GitHub using device flow."""
    github_login_impl()


@github_app.command("logout")
def github_logout_cmd() -> None:
    """Remove stored GitHub credentials."""
    github_logout_impl()


@git_app.command("login", hidden=True)
def git_login_cmd() -> None:
    """Deprecated alias for `potpie github login`."""
    github_login_impl()


@git_app.command("logout", hidden=True)
def git_logout_cmd() -> None:
    """Deprecated alias for `potpie github logout`."""
    github_logout_impl()


def github_repos_impl() -> None:
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
            "Run: potpie github login",
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


@github_app.command("repos")
def github_repos_cmd() -> None:
    """List GitHub repositories you can access."""
    github_repos_impl()


@github_test_app.command("repos", hidden=True)
@git_test_app.command("repos", hidden=True)
def github_test_repos_cmd() -> None:
    """Deprecated alias for ``potpie github repos``."""
    github_repos_impl()


def _build_auth_compat_github() -> typer.Typer:
    app = typer.Typer(help="[Deprecated] use `potpie github`.")
    app.command("login")(github_login_cmd)
    app.command("logout")(github_logout_cmd)
    app.command("repos")(github_repos_cmd)
    return app


github_app.add_typer(github_test_app, name="test")
git_app.add_typer(git_test_app, name="test")


def register_github_commands(root_app: typer.Typer) -> None:
    """Deprecated: use ``register_integration_commands``."""
    from adapters.inbound.cli.auth.auth_commands import register_integration_commands

    register_integration_commands(root_app)
