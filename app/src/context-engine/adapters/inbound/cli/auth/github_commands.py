"""GitHub CLI authentication commands (device flow)."""

from __future__ import annotations

import webbrowser

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
from adapters.inbound.cli.auth.auth_commands import register_provider_app
from adapters.inbound.cli.commands._common import EXIT_AUTH, EXIT_UNAVAILABLE, get_store
from adapters.outbound.cli_auth.credentials_store import (
    CredentialStoreError,
    ProviderCredentialError,
)
from adapters.inbound.cli.ui.output import emit_error, print_json_blob, print_plain_line

github_app = typer.Typer(help="GitHub authentication.")
github_test_app = typer.Typer(help="GitHub test helpers.")
git_app = typer.Typer(
    help="Deprecated: use `potpie auth github` instead.",
    hidden=True,
)
git_test_app = typer.Typer(help="Git provider test helpers.", hidden=True)


def _flags() -> tuple[bool, bool]:
    from adapters.inbound.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


def _open_github_device_verification(user_code: str, verification_uri: str) -> None:
    print_plain_line(
        "GitHub login requires a one-time verification code.",
        as_json=False,
    )
    print_plain_line(f"Copy this code: {user_code}", as_json=False)
    print_plain_line(
        "Copy the code, then press Enter to open GitHub.",
        as_json=False,
    )
    typer.confirm("Press Enter after copying the code", default=True, show_default=False)
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


def github_login_impl() -> None:
    """Authenticate the CLI with GitHub using device flow."""
    load_cli_env()
    j, v = _flags()
    store = get_store()
    account = None
    payload = None
    try:
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


def github_logout_impl() -> None:
    """Remove GitHub credentials from keychain and config."""
    j, v = _flags()
    store = get_store()
    try:
        store.clear_provider_credentials("github")
    except (ProviderCredentialError, CredentialStoreError, ValueError) as exc:
        emit_error("GitHub logout failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc

    if j:
        print_json_blob({"ok": True, "provider": "github"}, as_json=True)
        return
    print_plain_line("Logged out of GitHub.", as_json=False)


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
    """Deprecated alias for `potpie auth github login`."""
    github_login_impl()


@git_app.command("logout", hidden=True)
def git_logout_cmd() -> None:
    """Deprecated alias for `potpie auth github logout`."""
    github_logout_impl()


@github_test_app.command("repos")
@git_test_app.command("repos")
def github_test_repos_cmd() -> None:
    """List GitHub repositories owned by the authenticated user."""
    j, v = _flags()
    store = get_store()
    try:
        credentials = store.get_provider_credentials("github")
    except (ProviderCredentialError, CredentialStoreError) as exc:
        emit_error(
            "GitHub credentials not found",
            str(exc),
            verbose=v,
        )
        raise typer.Exit(code=EXIT_AUTH) from exc
    token = str(credentials.get("access_token") or "").strip()
    if not token:
        emit_error(
            "GitHub credentials not found",
            "GitHub token not found in system keychain. Run: potpie auth github login",
            verbose=v,
        )
        raise typer.Exit(code=EXIT_AUTH)
    try:
        repos = list_user_owned_repositories(token)
    except GitHubDeviceFlowError as exc:
        emit_error("GitHub repository listing failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc

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


def register_github_commands(root_app: typer.Typer) -> None:
    """Wire GitHub auth and test commands into the root CLI."""
    register_provider_app("github", github_app)
    github_root = typer.Typer(
        help="GitHub CLI helpers (use `potpie auth github` to sign in).",
    )
    github_root.add_typer(github_test_app, name="test")
    root_app.add_typer(github_root, name="github")
    git_app.add_typer(git_test_app, name="test")
    root_app.add_typer(git_app, name="git")
