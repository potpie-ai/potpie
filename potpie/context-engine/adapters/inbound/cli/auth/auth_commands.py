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

from adapters.inbound.cli.auth.atlassian_auth import run_atlassian_api_token_auth
from adapters.inbound.cli.auth.atlassian_read import (
    AtlassianReadError,
    fetch_confluence_content_sample,
    fetch_confluence_spaces_sample,
    fetch_jira_issues_sample,
    fetch_jira_projects,
    run_confluence_use_flow,
    run_jira_use_flow,
)
from adapters.outbound.cli_auth.callback_server import (
    OAuthCallbackResult,
    wait_for_oauth_callback,
)
from adapters.outbound.cli_auth.credentials_store import (
    ProviderCredentialError,
    credentials_path,
    get_integration_status,
    get_integration_tokens,
)
from adapters.outbound.cli_auth.env_bootstrap import load_cli_env
from adapters.outbound.cli_auth.integration_session import (
    ensure_valid_integration_tokens,
    token_needs_refresh,
)
from adapters.outbound.cli_auth.integration_verify import verify_integration_access
from adapters.inbound.cli.commands._common import EXIT_AUTH, EXIT_UNAVAILABLE, get_store
from adapters.inbound.cli.ui.output import emit_error, print_json_blob, print_plain_line
from adapters.outbound.cli_auth.pkce import generate_pkce_pair
from adapters.outbound.cli_auth.provider_config import (
    Provider,
    authorization_url,
    get_callback_host,
    get_callback_path,
    get_callback_port,
    get_client_id,
    get_redirect_uri,
    get_scopes,
)
from adapters.outbound.cli_auth.token_exchange import exchange_authorization_code

auth_app = typer.Typer(help="Authenticate CLI integrations.")
linear_app = typer.Typer(help="Linear authentication.")
jira_app = typer.Typer(help="Jira authentication and read.")
confluence_app = typer.Typer(help="Confluence authentication and read.")

_OAUTH_CALLBACK_TIMEOUT = 300.0
_ALL_PROVIDERS: tuple[str, ...] = ("linear", "jira", "confluence")


def _canonical_provider_for_json(product: str) -> str:
    """Map CLI product aliases to provider keys used in JSON output."""
    key = product.strip().lower()
    if key in {"wiki", "conf", "confluence"}:
        return "confluence"
    return key


def register_provider_app(name: str, provider_app: typer.Typer) -> None:
    """Register a provider-specific auth sub-application."""
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("provider app name must be non-empty")
    auth_app.add_typer(provider_app, name=key)


def _flags() -> tuple[bool, bool]:
    from adapters.inbound.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


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
    store = get_store()

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
        raise typer.Exit(code=EXIT_AUTH) from callback_error

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
        raise typer.Exit(code=EXIT_AUTH) from callback_error

    if callback_result is None:
        emit_error(
            "OAuth callback timed out",
            f"No response on {redirect_uri} within {_OAUTH_CALLBACK_TIMEOUT:.0f}s.",
            verbose=v,
        )
        raise typer.Exit(code=EXIT_AUTH)

    callback = callback_result
    if callback.error:
        msg = callback.error
        if callback.error_description:
            msg = f"{msg}: {callback.error_description}"
        emit_error("Linear OAuth failed", msg, verbose=v)
        raise typer.Exit(code=EXIT_AUTH)

    if not callback.code:
        emit_error(
            "Linear OAuth failed",
            "No authorization code received.",
            verbose=v,
        )
        raise typer.Exit(code=EXIT_AUTH)

    if callback.state != state:
        emit_error(
            "Linear OAuth failed",
            "State mismatch; aborting for safety.",
            verbose=v,
        )
        raise typer.Exit(code=EXIT_AUTH)

    try:
        tokens = exchange_authorization_code(
            "linear",
            code=callback.code,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
        )
        store.save_integration_tokens("linear", tokens)
    except (ValueError, RuntimeError, ProviderCredentialError) as exc:
        emit_error("Linear token exchange failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc

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
                if provider == "linear":
                    credentials = ensure_valid_integration_tokens(provider)
                else:
                    credentials = get_integration_tokens(provider)
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
        help="Provider to log out: linear, jira, or confluence.",
    ),
) -> None:
    """Remove locally stored credentials for a provider."""
    load_cli_env()
    j, v = _flags()
    store = get_store()
    key = provider.strip().lower()
    if key in {"wiki", "conf"}:
        key = "confluence"
    if key not in _ALL_PROVIDERS:
        emit_error(
            "Unknown provider",
            f"Expected one of: {', '.join(_ALL_PROVIDERS)}.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    existing = get_integration_status(key)
    was_authenticated = bool(existing.get("authenticated"))

    if key == "linear" and not was_authenticated:
        emit_error(
            f"{key} not authenticated",
            "No stored credentials to revoke.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    try:
        store.clear_integration_tokens(key)
    except (ProviderCredentialError, ValueError) as exc:
        emit_error(f"{key} logout failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc

    if was_authenticated:
        message = f"Logged out of {key}."
    else:
        message = f"No active {key} session; removed any stale local credentials."
    payload: dict[str, Any] = {"ok": True, "provider": key}
    if not was_authenticated:
        payload["cleared_stale"] = True
    print_plain_line(message, as_json=j, json_payload=payload)


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


def _print_jira_issue_row(row: dict[str, Any]) -> None:
    _print_remote_line(
        f"\n{_esc(row.get('key'))}  {_esc(row.get('summary'))}".rstrip(),
    )
    if row.get("project"):
        _print_remote_line(f"  Project: {_esc(row.get('project'))}")
    if row.get("status"):
        _print_remote_line(f"  Status: {_esc(row.get('status'))}")
    if row.get("type"):
        _print_remote_line(f"  Type: {_esc(row.get('type'))}")
    if row.get("assignee"):
        _print_remote_line(f"  Assignee: {_esc(row.get('assignee'))}")
    if row.get("priority"):
        _print_remote_line(f"  Priority: {_esc(row.get('priority'))}")
    if row.get("created"):
        created_line = f"  Created: {_esc(row.get('created'))}"
        if row.get("reporter"):
            created_line = f"{created_line} · {_esc(row.get('reporter'))}"
        _print_remote_line(created_line)
    if row.get("updated"):
        _print_remote_line(f"  Updated: {_esc(row.get('updated'))}")
    if row.get("description"):
        _print_remote_line(f"  Description: {_esc(row.get('description'))}")
    if row.get("url"):
        _print_remote_line(f"  URL: {_esc(row.get('url'))}")


def _print_wiki_row(row: dict[str, Any], *, pages: bool) -> None:
    if pages:
        _print_remote_line(f"\n{_esc(row.get('title'))}")
        if row.get("space"):
            _print_remote_line(f"  Space: {_esc(row.get('space'))}")
        if row.get("status"):
            _print_remote_line(f"  Status: {_esc(row.get('status'))}")
        if row.get("created"):
            created_line = f"  Created: {_esc(row.get('created'))}"
            if row.get("created_by"):
                created_line = f"{created_line} · {_esc(row.get('created_by'))}"
            _print_remote_line(created_line)
        if row.get("updated"):
            updated_line = f"  Updated: {_esc(row.get('updated'))}"
            if row.get("updated_by"):
                updated_line = f"{updated_line} · {_esc(row.get('updated_by'))}"
            _print_remote_line(updated_line)
        if row.get("excerpt"):
            _print_remote_line(f"  Excerpt: {_esc(row.get('excerpt'))}")
        if row.get("url"):
            _print_remote_line(f"  URL: {_esc(row.get('url'))}")
        return
    line = f"  {_esc(row.get('key'))}\t{_esc(row.get('name'))}\t{_esc(row.get('type'))}"
    _print_remote_line(line)
    if row.get("url"):
        _print_remote_line(f"    {_esc(row.get('url'))}")


def _run_atlassian_quick_read(
    *,
    product: str,
    fetcher,
    limit: int,
    hint_cmd: str = "potpie auth jira select",
    display_name: str | None = None,
) -> None:
    load_cli_env()
    j, v = _flags()
    name = display_name or ("Confluence" if product == "wiki" else product.capitalize())
    try:
        rows = fetcher(limit=limit)
    except AtlassianReadError as exc:
        emit_error(f"{name} read failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc

    if j:
        print_json_blob(
            {
                "ok": True,
                "provider": "jira" if product == "jira" else "confluence",
                "product": product,
                "count": len(rows),
                "items": rows,
            },
            as_json=True,
        )
        return

    pages = product == "wiki" and rows and "title" in (rows[0] or {})
    row_label = "pages" if pages else ("issues" if product == "jira" else "spaces")
    print_plain_line(f"{name} ({len(rows)} {row_label}):", as_json=False)
    if not rows:
        print_plain_line(f"  (no results — run: {hint_cmd})", as_json=False)
        return
    for row in rows:
        if product == "jira":
            if row.get("description") or row.get("url"):
                _print_jira_issue_row(row)
            else:
                _print_remote_line(
                    f"  {_esc(row.get('key'))}\t{_esc(row.get('status'))}"
                    f"\t{_esc(row.get('project'))}\t{_esc(row.get('summary'))}",
                )
        else:
            _print_wiki_row(row, pages=pages)


def _run_product_use_result(
    result: dict[str, Any],
    *,
    product_label: str,
) -> None:
    load_cli_env()
    j, _ = _flags()
    if j:
        provider = _canonical_provider_for_json(str(result.get("product") or ""))
        print_json_blob(
            {"ok": True, **result, "provider": provider},
            as_json=True,
        )
        return
    _print_remote_line(
        f"{product_label} workspace: {_esc(result.get('workspace_key'))} "
        f"({_esc(result.get('workspace_name'))})",
    )
    rows = result.get("items") or []
    print_plain_line(f"{len(rows)} item(s):", as_json=False)
    for row in rows:
        if result["product"] == "jira":
            _print_jira_issue_row(row)
        else:
            _print_wiki_row(row, pages=True)


@jira_app.command("login")
def jira_login(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-authenticate even if Jira is already connected.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        help="Atlassian account email (non-interactive login).",
    ),
    api_token: str | None = typer.Option(
        None,
        "--api-token",
        help="Atlassian API token (non-interactive login).",
    ),
    site_subdomain: str | None = typer.Option(
        None,
        "--site-subdomain",
        help="Atlassian site subdomain, e.g. myteam for myteam.atlassian.net.",
    ),
) -> None:
    """Connect Jira with an Atlassian API token (Jira access only)."""
    load_cli_env()
    j, v = _flags()
    run_atlassian_api_token_auth(
        "jira",
        force=force,
        as_json=j,
        verbose=v,
        email=email,
        api_token=api_token,
        site_subdomain=site_subdomain,
    )


@jira_app.command("logout")
def jira_logout() -> None:
    """Remove stored Jira credentials."""
    auth_logout("jira")


@jira_app.command("ls")
def jira_ls(
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=50),
) -> None:
    """List Jira projects you can access."""
    load_cli_env()
    j, v = _flags()
    try:
        rows = fetch_jira_projects(limit=limit)
    except AtlassianReadError as exc:
        emit_error("Jira workspace list failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    if j:
        print_json_blob(
            {"ok": True, "provider": "jira", "projects": rows}, as_json=True
        )
        return
    print_plain_line("Jira projects:", as_json=False)
    if not rows:
        print_plain_line("  (none)", as_json=False)
    for row in rows:
        _print_remote_line(
            f"  {_esc(row.get('key'))}\t{_esc(row.get('name'))}\t{_esc(row.get('type'))}",
        )
        if row.get("url"):
            _print_remote_line(f"    {_esc(row.get('url'))}")
    print_plain_line("\nFetch issues: potpie auth jira select", as_json=False)


@jira_app.command("select")
def jira_select(
    key: str | None = typer.Option(None, "--key", "-k", help="Jira project key."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
) -> None:
    """Select a Jira project and fetch issues in the terminal."""
    load_cli_env()
    j, v = _flags()
    try:
        result = run_jira_use_flow(workspace_key=key, limit=limit)
    except AtlassianReadError as exc:
        emit_error("Jira fetch failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    _run_product_use_result(result, product_label="Jira")


@jira_app.command("issues")
def jira_issues(
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
) -> None:
    """Fetch recent Jira issues (saved project, or first project)."""
    _run_atlassian_quick_read(
        product="jira",
        fetcher=fetch_jira_issues_sample,
        limit=limit,
        hint_cmd="potpie auth jira select",
        display_name="Jira",
    )


@confluence_app.command("login")
def confluence_login(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-authenticate even if Confluence is already connected.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        help="Atlassian account email (non-interactive login).",
    ),
    api_token: str | None = typer.Option(
        None,
        "--api-token",
        help="Atlassian API token (non-interactive login).",
    ),
    site_subdomain: str | None = typer.Option(
        None,
        "--site-subdomain",
        help="Atlassian site subdomain, e.g. myteam for myteam.atlassian.net.",
    ),
) -> None:
    """Connect Confluence with an Atlassian API token (Confluence access only)."""
    load_cli_env()
    j, v = _flags()
    run_atlassian_api_token_auth(
        "confluence",
        force=force,
        as_json=j,
        verbose=v,
        email=email,
        api_token=api_token,
        site_subdomain=site_subdomain,
    )


@confluence_app.command("logout")
def confluence_logout() -> None:
    """Remove stored Confluence credentials."""
    auth_logout("confluence")


@confluence_app.command("ls")
def confluence_ls(
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=50),
) -> None:
    """List Confluence spaces you can access."""
    load_cli_env()
    j, v = _flags()
    try:
        rows = fetch_confluence_spaces_sample(limit=limit)
    except AtlassianReadError as exc:
        emit_error("Confluence workspace list failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    if j:
        print_json_blob(
            {"ok": True, "provider": "confluence", "spaces": rows}, as_json=True
        )
        return
    print_plain_line("Confluence spaces:", as_json=False)
    if not rows:
        print_plain_line("  (none)", as_json=False)
    for row in rows:
        _print_remote_line(
            f"  {_esc(row.get('key'))}\t{_esc(row.get('name'))}\t{_esc(row.get('type'))}",
        )
    print_plain_line("\nFetch pages: potpie auth confluence select", as_json=False)


@confluence_app.command("select")
def confluence_select(
    key: str | None = typer.Option(None, "--key", "-k", help="Confluence space key."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
) -> None:
    """Select a Confluence space and fetch pages in the terminal."""
    load_cli_env()
    j, v = _flags()
    try:
        result = run_confluence_use_flow(workspace_key=key, limit=limit)
    except AtlassianReadError as exc:
        emit_error("Confluence fetch failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_UNAVAILABLE) from exc
    _run_product_use_result(result, product_label="Confluence")


@confluence_app.command("pages")
def confluence_pages(
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
) -> None:
    """Fetch Confluence pages (saved space) or list spaces if none saved."""
    _run_atlassian_quick_read(
        product="wiki",
        fetcher=fetch_confluence_content_sample,
        limit=limit,
        hint_cmd="potpie auth confluence select",
        display_name="Confluence",
    )


register_provider_app("jira", jira_app)
register_provider_app("confluence", confluence_app)
