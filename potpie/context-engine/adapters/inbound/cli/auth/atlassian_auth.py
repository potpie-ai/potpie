"""Interactive Atlassian (Jira/Confluence) API-token login command.

The HTTP client (verification + site discovery) lives in
``adapters.outbound.cli_auth.atlassian_client``; this inbound module owns the
interactive flow (prompts, output, exit codes) and re-exports the client symbols
for callers that historically imported them from here.
"""

from __future__ import annotations

import sys
import time
import webbrowser
from collections.abc import Callable
from typing import Any

import typer

import click

from adapters.outbound.cli_auth.atlassian_client import (  # noqa: F401  (re-export)
    AtlassianAuthErrorKind,
    AtlassianAuthScheme,
    AtlassianVerifyResult,
    _auth_header_variants,
    _candidate_subdomains_from_email,
    _classify_gateway_status,
    _fetch_accessible_resources,
    _finalize_atlassian_site_unscoped,
    _finalize_selected_site,
    _gateway_base_url,
    _gateway_probe_paths,
    _merge_site_candidate,
    _parse_accessible_resources,
    _parse_gateway_probe_success,
    _parse_profile_name,
    _resolve_site_from_subdomain,
    _site_record,
    atlassian_basic_auth_header,
    atlassian_bearer_auth_header,
    collect_login_site_candidates,
    collect_site_candidates,
    discover_sites_with_api_token,
    fetch_accessible_resources,
    fetch_cloud_id_for_site,
    normalize_site_url,
    site_url_from_subdomain,
    token_style_from_succeeded_scheme,
    verify_gateway_product,
    verify_site_with_api_token,
)
from adapters.outbound.cli_auth.credentials_store import (
    ProviderCredentialError,
    credentials_path,
    get_integration_status,
    integration_token_storage,
)
from adapters.inbound.cli.commands._common import EXIT_AUTH, get_store
from adapters.inbound.cli.ui.output import emit_error, print_plain_line
from adapters.outbound.cli_auth.provider_config import (
    ATLASSIAN_API_TOKEN_PAGE,
    AtlassianProduct,
)


def _guard_typer_prompt[T](callback: Callable[[], T]) -> T:
    """Map Click/Typer Ctrl+C aborts to ``KeyboardInterrupt`` for callers."""
    try:
        return callback()
    except click.Abort:
        raise KeyboardInterrupt from None


def _prompt_site_subdomain() -> str:
    return _guard_typer_prompt(
        lambda: typer.prompt(
            "Enter your Atlassian site subdomain "
            "(e.g. 'potpie-team' for potpie-team.atlassian.net)"
        ).strip()
    )


def _prompt_and_resolve_site() -> tuple[
    dict[str, Any] | None, AtlassianAuthErrorKind | None
]:
    """Prompt once for site subdomain and resolve cloud ID."""
    return _resolve_site_from_subdomain(_prompt_site_subdomain())


def _cli_credentials_supplied(
    email: str | None,
    api_token: str | None,
    site_subdomain: str | None,
) -> bool:
    return bool(
        (email or "").strip()
        and (api_token or "").strip()
        and (site_subdomain or "").strip()
    )


def _prompt_credentials() -> tuple[str, str]:
    api_token = _guard_typer_prompt(
        lambda: typer.prompt("Enter your API token", hide_input=True).strip()
    )
    if not api_token:
        raise typer.Exit(code=1)
    email = _guard_typer_prompt(
        lambda: typer.prompt("Enter your Atlassian email").strip()
    )
    if not email:
        raise typer.Exit(code=1)
    return email, api_token


def _open_atlassian_api_token_page(product: AtlassianProduct) -> None:
    """Show setup steps, then open the Atlassian token page after confirmation."""
    product_label = "Jira" if product == "jira" else "Confluence"
    print_plain_line(
        f"{product_label} login — Atlassian API token",
        as_json=False,
    )
    for line in (
        "  • Create a token at id.atlassian.com (without scopes)",
        "  • One token works for both Jira and Confluence",
        "  • Press Enter to open the page, then paste the token with your email and site",
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

    print_plain_line("Opening id.atlassian.com ...", as_json=False)

    opened = webbrowser.open(ATLASSIAN_API_TOKEN_PAGE, new=1)
    if not opened:
        print_plain_line(
            "Could not open a browser. Open this URL:",
            as_json=False,
        )
        print_plain_line(ATLASSIAN_API_TOKEN_PAGE, as_json=False, markup=False)
        return
    print_plain_line("Paste the token below when you are ready.", as_json=False)


def _auth_failure_message(
    product: AtlassianProduct,
    error_kind: AtlassianAuthErrorKind | None = None,
) -> str:
    name = product.capitalize()
    lines = [
        f"Could not authenticate {name} with Atlassian.",
        "  - Use an API token from id.atlassian.com (scoped tokens are supported)",
        "  - Email must match the Atlassian account that created the token",
        "  - Your account must have access to the product on the selected site",
    ]
    lines.append(
        "  - Use Create API token (without scopes) for Jira + Confluence on one token"
    )
    lines.append(
        "  - Or use scoped tokens: Jira needs read:jira-work; "
        "Confluence needs read/content scopes (one product per token)"
    )
    if error_kind == AtlassianAuthErrorKind.INVALID_CREDENTIALS:
        lines.insert(1, "  - Invalid email or API token")
    elif error_kind == AtlassianAuthErrorKind.INSUFFICIENT_SCOPES:
        if product == "confluence":
            scope_hint = (
                "  - Token is missing required read scopes "
                "(Confluence: read:confluence-content at minimum)"
            )
        else:
            scope_hint = (
                "  - Token is missing required read scopes "
                "(Jira: read:jira-work at minimum)"
            )
        lines.insert(1, scope_hint)
    elif error_kind == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED:
        lines.insert(1, "  - Could not resolve cloud ID for the site")
    elif error_kind == AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED:
        lines.insert(
            1,
            f"  - Token or account cannot access {name} on this site",
        )
    else:
        lines.insert(
            1,
            "  - Check the site subdomain from your Jira URL (e.g. acme.atlassian.net)",
        )
    return "\n".join(lines)


def _get_product_credentials(product: AtlassianProduct) -> dict[str, Any]:
    store = get_store()
    if product == "jira":
        return store.get_jira_credentials()
    return store.get_confluence_credentials()


def _save_product_credentials(
    product: AtlassianProduct, payload: dict[str, Any]
) -> None:
    store = get_store()
    if product == "jira":
        store.save_jira_credentials(payload)
    else:
        store.save_confluence_credentials(payload)


def run_atlassian_api_token_auth(
    product: AtlassianProduct,
    *,
    force: bool = False,
    as_json: bool = False,
    verbose: bool = False,
    email: str | None = None,
    api_token: str | None = None,
    site_subdomain: str | None = None,
) -> None:
    """Authenticate Jira or Confluence using an Atlassian API token (one product only)."""
    product_label = product.capitalize()
    try:
        existing = _get_product_credentials(product)
    except ProviderCredentialError as exc:
        emit_error(
            f"{product_label} credential lookup failed", str(exc), verbose=verbose
        )
        existing = {}

    if existing.get("api_token") and existing.get("site_url") and not force:
        print_plain_line(
            f"{product_label} is already connected.",
            as_json=as_json,
            json_payload={
                "ok": True,
                "already_connected": True,
                "provider": product,
                "site_url": existing.get("site_url"),
                "site_name": existing.get("site_name"),
                "cloud_id": existing.get("cloud_id"),
            },
        )
        return

    supplied = _cli_credentials_supplied(email, api_token, site_subdomain)
    if not sys.stdin.isatty() and not supplied:
        emit_error(
            f"{product_label} authentication requires a terminal",
            "Run in an interactive shell to enter email, API token, and site subdomain, "
            "or pass all three when invoking login non-interactively.",
            verbose=verbose,
        )
        raise typer.Exit(code=1)

    if supplied:
        email_value = (email or "").strip()
        api_token_value = (api_token or "").strip()
        site, last_error = _resolve_site_from_subdomain(site_subdomain or "")
    else:
        if not as_json:
            _open_atlassian_api_token_page(product)
        else:
            webbrowser.open(ATLASSIAN_API_TOKEN_PAGE, new=1)
        email_value, api_token_value = _prompt_credentials()
        site, last_error = _prompt_and_resolve_site()
    email, api_token = email_value, api_token_value
    if not site:
        emit_error(
            f"{product.capitalize()} authentication failed",
            _auth_failure_message(product, last_error),
            verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH)

    site, last_error = _finalize_selected_site(email, api_token, site, product)
    if not site:
        emit_error(
            f"{product_label} authentication failed",
            _auth_failure_message(product, last_error),
            verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH)

    token_style = str(site.get("token_style") or "").strip() or "classic"
    payload = {
        "auth_type": "api_token",
        "token_style": token_style,
        "email": email,
        "api_token": api_token,
        "cloud_id": str(site.get("cloud_id") or "").strip(),
        "site_url": site["site_url"],
        "site_name": site["site_name"],
        "stored_at": time.time(),
    }
    try:
        _save_product_credentials(product, payload)
    except ProviderCredentialError as exc:
        emit_error(
            f"{product_label} credential storage failed", str(exc), verbose=verbose
        )
        raise typer.Exit(code=EXIT_AUTH) from exc

    token_storage = integration_token_storage()
    stored = get_integration_status(product)
    if stored.get("token_storage"):
        token_storage = str(stored["token_storage"])
    storage_label = (
        "system keychain"
        if token_storage == "keychain"
        else "local credentials file"
    )
    summary = (
        f"Connected {product_label} to {site['site_url']}. "
        f"Stored tokens in {storage_label}; metadata saved to {credentials_path()}."
    )
    print_plain_line(
        summary,
        as_json=as_json,
        json_payload={
            "ok": True,
            "provider": product,
            "token_style": token_style,
            "site_url": site["site_url"],
            "site_name": site["site_name"],
            "cloud_id": payload["cloud_id"],
            "path": str(credentials_path()),
            "token_storage": token_storage,
            "product_verified": product,
        },
    )
