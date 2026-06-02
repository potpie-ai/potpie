"""CLI commands for external integration authentication."""

from __future__ import annotations

import secrets
import threading
import time
import urllib.parse
import webbrowser
from typing import Any

import typer
from rich.markup import escape

from adapters.inbound.cli.callback_server import OAuthCallbackResult, wait_for_oauth_callback
from adapters.inbound.cli.credentials_store import (
    ProviderCredentialError,
    clear_integration_tokens,
    credentials_path,
    get_integration_status,
    save_integration_tokens,
)
from adapters.inbound.cli.env_bootstrap import load_cli_env
from adapters.inbound.cli.integration_session import (
    ensure_valid_integration_tokens,
    token_needs_refresh,
)
from adapters.inbound.cli.integration_verify import verify_integration_access
from adapters.inbound.cli.output import emit_error, print_json_blob, print_plain_line
from adapters.inbound.cli.pkce import generate_pkce_pair
from adapters.inbound.cli.provider_config import (
    Provider,
    authorization_url,
    get_callback_host,
    get_callback_path,
    get_callback_port,
    get_client_id,
    get_redirect_uri,
    get_scopes,
)
from adapters.inbound.cli.token_exchange import exchange_authorization_code

auth_app = typer.Typer(help="Authenticate CLI integrations.")
linear_app = typer.Typer(help="Linear authentication.")

_OAUTH_CALLBACK_TIMEOUT = 300.0
_ALL_PROVIDERS: tuple[str, ...] = ("linear",)


def register_provider_app(name: str, provider_app: typer.Typer) -> None:
    """Register a provider-specific auth sub-application."""
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("provider app name must be non-empty")
    auth_app.add_typer(provider_app, name=key)


def _flags() -> tuple[bool, bool]:
    from adapters.inbound.cli.main import _flags as main_flags

    return main_flags()


def _build_linear_authorization_url(
    *,
    redirect_uri: str,
    state: str,
    code_challenge: str,
) -> str:
    params = {
        "client_id": get_client_id("linear"),
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": get_scopes("linear"),
        "state": state,
        "prompt": "consent",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f"{authorization_url('linear')}?{query}"


def _wait_for_callback(*, host: str, port: int, path: str) -> OAuthCallbackResult:
    return wait_for_oauth_callback(
        host=host,
        port=port,
        path=path,
        timeout=_OAUTH_CALLBACK_TIMEOUT,
    )


def _token_is_expired(expires_at: Any) -> bool:
    if expires_at is None:
        return False
    try:
        return time.time() > float(expires_at)
    except (TypeError, ValueError):
        return False


def _try_refresh_linear_session() -> bool:
    try:
        tokens = ensure_valid_integration_tokens("linear")
    except (ValueError, RuntimeError, ProviderCredentialError):
        return False
    return bool(tokens.get("access_token")) and not _token_is_expired(
        tokens.get("expires_at")
    )


def _handle_already_connected(provider: Provider, status: dict[str, Any]) -> None:
    j, _ = _flags()
    name = provider.capitalize()
    print_plain_line(
        f"{name} is already connected.",
        as_json=j,
        json_payload={
            "ok": True,
            "already_connected": True,
            "provider": provider,
            "auth_type": status.get("auth_type"),
            "expires_at": status.get("expires_at"),
            "cloud_id": status.get("cloud_id"),
            "site_url": status.get("site_url"),
            "site_name": status.get("site_name"),
            "token_storage": status.get("token_storage"),
        },
    )


def _print_linear_login_success(
    status: dict[str, Any],
    *,
    tokens: dict[str, Any] | None = None,
    refreshed: bool = False,
) -> None:
    """Emit one human line or one JSON document for a successful Linear login."""
    j, _ = _flags()
    account = (
        status.get("login"),
        status.get("email"),
        status.get("site_name"),
    )
    who = account[0] or account[1] or "Linear"
    org_suffix = f" @ {account[2]}" if account[2] else ""
    if refreshed:
        summary = f"Linear session refreshed for {who}{org_suffix}."
    else:
        summary = (
            f"Logged in to Linear as {who}{org_suffix}. "
            f"Stored tokens in system keychain; metadata saved to {credentials_path()}."
        )
    if not j:
        print_plain_line(summary, as_json=False)
        return
    payload: dict[str, Any] = {
        "ok": True,
        "provider": "linear",
        "path": str(credentials_path()),
        "login": status.get("login"),
        "email": status.get("email"),
        "organization": status.get("site_name"),
        "token_storage": "keychain",
        "auth_type": status.get("auth_type"),
        "expires_at": status.get("expires_at"),
    }
    if refreshed:
        payload["refreshed"] = True
    if tokens:
        payload["scope"] = tokens.get("scope")
    print_json_blob(payload, as_json=True)


def _run_linear_oauth_flow(*, force: bool = False) -> None:
    load_cli_env()
    j, v = _flags()

    status = get_integration_status("linear")
    if status.get("authenticated") and not force:
        if token_needs_refresh(status.get("expires_at")):
            if _try_refresh_linear_session():
                _print_linear_login_success(
                    get_integration_status("linear"),
                    refreshed=True,
                )
                return
            if not j:
                print_plain_line(
                    "Linear session expired; re-authenticating...",
                    as_json=False,
                )
        else:
            _handle_already_connected("linear", status)
            return

    client_id = get_client_id("linear")
    if not client_id:
        emit_error(
            "Linear OAuth not configured",
            "Linear OAuth client id is missing (set LINEAR_CLIENT_ID in your environment).",
            verbose=v,
        )
        raise typer.Exit(code=1)

    try:
        redirect_uri = get_redirect_uri()
        port = get_callback_port()
        host = get_callback_host()
        callback_path = get_callback_path()
    except ValueError as exc:
        emit_error("Linear OAuth redirect is invalid", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    state = secrets.token_urlsafe(24)
    code_verifier, code_challenge = generate_pkce_pair()
    auth_url = _build_linear_authorization_url(
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )

    callback_result: OAuthCallbackResult | None = None
    callback_error: BaseException | None = None

    def _capture_callback() -> None:
        nonlocal callback_result, callback_error
        try:
            callback_result = _wait_for_callback(
                host=host,
                port=port,
                path=callback_path,
            )
        except BaseException as exc:
            callback_error = exc

    server_thread = threading.Thread(target=_capture_callback, daemon=True)
    server_thread.start()
    time.sleep(0.15)
    if callback_error is not None:
        emit_error(
            "OAuth callback failed to start",
            str(callback_error),
            verbose=v,
            exc=callback_error if v else None,
        )
        raise typer.Exit(code=1) from callback_error

    if not j:
        print_plain_line(
            "Opening browser for Linear authentication...",
            as_json=False,
        )
        print_plain_line(
            f"If the browser does not open, visit:\n{auth_url}",
            as_json=False,
        )

    opened = webbrowser.open(auth_url, new=1)
    if not opened and not j:
        print_plain_line(
            "Could not open a browser automatically; use the URL above.",
            as_json=False,
        )

    server_thread.join(timeout=_OAUTH_CALLBACK_TIMEOUT + 5.0)

    if callback_error is not None:
        if isinstance(callback_error, TimeoutError):
            emit_error("OAuth callback timed out", str(callback_error), verbose=v)
        else:
            emit_error(
                "OAuth callback failed",
                str(callback_error),
                verbose=v,
                exc=callback_error if v else None,
            )
        raise typer.Exit(code=1) from callback_error

    if callback_result is None:
        emit_error(
            "OAuth callback timed out",
            f"No response on {redirect_uri} within {_OAUTH_CALLBACK_TIMEOUT:.0f}s.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    callback = callback_result
    if callback.error:
        msg = callback.error
        if callback.error_description:
            msg = f"{msg}: {callback.error_description}"
        emit_error("Linear OAuth failed", msg, verbose=v)
        raise typer.Exit(code=1)

    if not callback.code:
        emit_error(
            "Linear OAuth failed",
            "No authorization code received.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    if callback.state != state:
        emit_error(
            "Linear OAuth failed",
            "State mismatch; aborting for safety.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    try:
        tokens = exchange_authorization_code(
            "linear",
            code=callback.code,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
        )
        save_integration_tokens("linear", tokens)
    except (ValueError, RuntimeError, ProviderCredentialError) as exc:
        emit_error("Linear token exchange failed", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    _print_linear_login_success(get_integration_status("linear"), tokens=tokens)


@auth_app.command("status")
def auth_status(
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Run a lightweight read-only API check for authenticated providers.",
    ),
) -> None:
    """Show local integration auth status."""
    load_cli_env()
    j, _ = _flags()
    rows: list[dict[str, Any]] = []

    for provider in _ALL_PROVIDERS:
        meta = get_integration_status(provider)
        row = dict(meta)
        if verify and meta.get("authenticated"):
            try:
                credentials = ensure_valid_integration_tokens(provider)
            except (ValueError, RuntimeError, ProviderCredentialError) as exc:
                row["verified"] = False
                row["verify_message"] = str(exc)
            else:
                ok, message = verify_integration_access(provider, credentials)
                row["verified"] = ok
                row["verify_message"] = message
                if provider == "linear":
                    refreshed = get_integration_status(provider)
                    row["expires_at"] = refreshed.get("expires_at")
        rows.append(row)

    if j:
        print_json_blob({"integrations": rows}, as_json=True)
        return

    for row in rows:
        provider = row["provider"]
        if not row.get("authenticated"):
            print_plain_line(f"{provider}: not authenticated", as_json=False)
            continue
        parts = [f"{provider}: authenticated"]
        if row.get("login"):
            parts.append(f"login={_esc(row['login'])}")
        if row.get("email"):
            parts.append(f"email={_esc(row['email'])}")
        if row.get("site_name"):
            parts.append(f"site={_esc(row['site_name'])}")
        if row.get("site_url"):
            parts.append(f"url={_esc(row['site_url'])}")
        if row.get("expires_at") is not None:
            parts.append(f"expires_at={_esc(row['expires_at'])}")
        if row.get("cloud_id"):
            parts.append(f"cloud_id={_esc(row['cloud_id'])}")
        if row.get("token_storage"):
            parts.append(f"token_storage={_esc(row['token_storage'])}")
        if verify:
            verified = row.get("verified")
            message = row.get("verify_message") or ""
            if verified is True:
                parts.append(f"verify={_esc(message)}")
            else:
                parts.append(f"verify failed ({_esc(message)})")
        _print_remote_line("  ".join(parts))


@auth_app.command("logout")
def auth_logout(
    provider: str = typer.Argument(
        ...,
        help="Provider to log out: linear.",
    ),
) -> None:
    """Remove locally stored credentials for a provider."""
    load_cli_env()
    j, v = _flags()
    key = provider.strip().lower()
    if key not in _ALL_PROVIDERS:
        emit_error(
            "Unknown provider",
            f"Expected one of: {', '.join(_ALL_PROVIDERS)}.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    existing = get_integration_status(key)
    if not existing.get("authenticated"):
        emit_error(
            f"{key} not authenticated",
            "No stored credentials to revoke.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    try:
        clear_integration_tokens(key)
    except (ProviderCredentialError, ValueError) as exc:
        emit_error(f"{key} logout failed", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    message = f"Logged out of {key}."
    print_plain_line(
        message,
        as_json=j,
        json_payload={"ok": True, "provider": key},
    )


@auth_app.command("revoke", hidden=True)
def auth_revoke(
    provider: str = typer.Argument(
        ...,
        help="Deprecated. Use `potpie auth logout <provider>`.",
    ),
) -> None:
    """Deprecated alias for `potpie auth logout`."""
    auth_logout(provider)




@auth_app.command("linear-login", hidden=True)
def auth_linear(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-authenticate even if Linear is already connected.",
    ),
) -> None:
    """Authenticate with Linear via OAuth (PKCE)."""
    _run_linear_oauth_flow(force=force)


def _esc(value: Any) -> str:
    """Escape provider-controlled text before Rich markup interpretation."""
    if value is None:
        return ""
    return escape(str(value))


def _print_remote_line(message: str) -> None:
    """Print provider-sourced text without Rich markup interpretation."""
    print_plain_line(message, as_json=False, markup=False)


@linear_app.command("login")
def linear_login(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-authenticate even if Linear is already connected.",
    ),
) -> None:
    """Authenticate with Linear via OAuth (PKCE)."""
    _run_linear_oauth_flow(force=force)


@linear_app.command("logout")
def linear_logout() -> None:
    """Remove stored Linear credentials."""
    auth_logout("linear")


register_provider_app("linear", linear_app)
