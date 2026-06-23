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
import select
from dataclasses import dataclass
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

AtlassianAuthTarget = Literal["atlassian", "jira", "confluence"]
T = TypeVar("T")


def _guard_typer_prompt(callback: Callable[[], T]) -> T:
    """Map Click/Typer Ctrl+C aborts to ``KeyboardInterrupt`` for callers."""
    try:
        return callback()
    except click.Abort:
        raise KeyboardInterrupt from None


@dataclass(frozen=True)
class ProductConnectResult:
    product: str
    status: str
    site_url: str | None = None
    cloud_id: str | None = None
    reason: str | None = None
    token_style: str | None = None


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


def _atlassian_target_label(product: AtlassianAuthTarget) -> str:
    if product == "atlassian":
        return "Atlassian"
    return product.capitalize()


def _open_atlassian_api_token_page(product: AtlassianAuthTarget) -> None:
    """Show setup steps, then open the Atlassian token page after confirmation."""
    product_label = _atlassian_target_label(product)
    print_plain_line(
        f"{product_label} login — Atlassian API token",
        as_json=False,
    )
    for line in (
        "  • Create a token at id.atlassian.com using API tokens without scopes",
        "  • One token works for both Jira and Confluence",
    ):
        print_plain_line(line, as_json=False)
    open_url_with_countdown(
        ATLASSIAN_API_TOKEN_PAGE,
        label="id.atlassian.com",
        timeout_seconds=10,
        open_message="Paste the token below when you are ready.",
    )


def open_url_with_countdown(
    url: str,
    *,
    label: str,
    timeout_seconds: int = 10,
    open_message: str | None = None,
) -> None:
    """Wait briefly so the user can read guidance, then open a browser or let Enter do it now."""
    print_plain_line(
        f"Press Enter to open {label} now, or wait {timeout_seconds}s for auto-open.",
        as_json=False,
    )
    try:
        if sys.stdin.isatty():
            for remaining in range(timeout_seconds, 0, -1):
                sys.stdout.write(f"\r  opening in {remaining}s ...")
                sys.stdout.flush()
                ready, _, _ = select.select([sys.stdin], [], [], 1.0)
                if ready:
                    try:
                        sys.stdin.readline()
                    except Exception:
                        pass
                    break
            sys.stdout.write("\n")
            sys.stdout.flush()
    except Exception:
        pass
    opened_ok = webbrowser.open(url, new=1)
    if not opened_ok:
        print_plain_line("Could not open a browser. Open this URL:", as_json=False)
        print_plain_line(url, as_json=False, markup=False)
    elif open_message:
        print_plain_line(open_message, as_json=False)


def _auth_failure_message(
    product: AtlassianAuthTarget,
    error_kind: AtlassianAuthErrorKind | None = None,
) -> str:
    name = _atlassian_target_label(product)
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
        elif product == "jira":
            scope_hint = (
                "  - Token is missing required read scopes "
                "(Jira: read:jira-work at minimum)"
            )
        else:
            scope_hint = (
                "  - Token is missing required read scopes for Jira or Confluence"
            )
        lines.insert(1, scope_hint)
    elif error_kind == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED:
        lines.insert(1, "  - Could not resolve cloud ID for the site")
    elif error_kind == AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED:
        product_hint = "Jira or Confluence" if product == "atlassian" else name
        lines.insert(
            1,
            f"  - Token or account cannot access {product_hint} on this site",
        )
    else:
        lines.insert(
            1,
            "  - Check the site subdomain from your Jira URL (e.g. acme.atlassian.net)",
        )
    return "\n".join(lines)


def _get_product_credentials(product: AtlassianAuthTarget) -> dict[str, Any]:
    store = get_store()
    if product == "jira":
        return store.get_jira_credentials()
    if product == "atlassian":
        return store.get_atlassian_credentials()
    return store.get_confluence_credentials()


def _save_product_credentials(
    product: AtlassianAuthTarget, payload: dict[str, Any]
) -> None:
    store = get_store()
    if product == "jira":
        store.save_jira_credentials(payload)
    elif product == "atlassian":
        store.save_atlassian_credentials(payload)
    else:
        store.save_confluence_credentials(payload)


def _result_reason(error_kind: AtlassianAuthErrorKind | None) -> str:
    if error_kind == AtlassianAuthErrorKind.INVALID_CREDENTIALS:
        return "invalid_credentials"
    if error_kind == AtlassianAuthErrorKind.INSUFFICIENT_SCOPES:
        return "insufficient_scopes"
    if error_kind == AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED:
        return "product_access_denied"
    return "unknown"


def connect_atlassian_product(
    product: AtlassianProduct,
    *,
    email: str,
    api_token: str,
    site_subdomain: str,
    force: bool = False,
    resolved_site: dict[str, Any] | None = None,
) -> ProductConnectResult:
    existing = _get_product_credentials(product)
    if existing.get("api_token") and existing.get("site_url") and not force:
        return ProductConnectResult(
            product=product,
            status="already_connected",
            site_url=existing.get("site_url"),
            cloud_id=existing.get("cloud_id"),
        )

    site = dict(resolved_site or {})
    last_error: AtlassianAuthErrorKind | None = None
    if not site:
        site, last_error = _resolve_site_from_subdomain(site_subdomain or "")
    if not site:
        return ProductConnectResult(
            product=product,
            status="not_connected",
            reason=_result_reason(last_error),
        )
    site, last_error = _finalize_selected_site(email, api_token, site, product)
    if not site:
        return ProductConnectResult(
            product=product,
            status="not_connected",
            reason=_result_reason(last_error),
        )
    token_style = str(site.get("token_style") or "").strip() or "classic"
    payload = {
        "auth_type": "api_token",
        "token_style": token_style,
        "email": email,
        "api_token": api_token,
        "cloud_id": str(site.get("cloud_id") or "").strip(),
        "site_url": site["site_url"],
        "site_name": site.get("site_name") or site["site_url"],
        "stored_at": time.time(),
    }
    _save_product_credentials(product, payload)
    return ProductConnectResult(
        product=product,
        status="connected",
        site_url=payload["site_url"],
        cloud_id=payload["cloud_id"],
        token_style=token_style,
    )


def run_atlassian_api_token_auth(
    product: AtlassianAuthTarget,
    *,
    force: bool = False,
    as_json: bool = False,
    verbose: bool = False,
    email: str | None = None,
    api_token: str | None = None,
    site_subdomain: str | None = None,
) -> None:
    """Authenticate an Atlassian target using an Atlassian API token."""
    product_label = _atlassian_target_label(product)
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

    try:
        result = connect_atlassian_product(
            product,
            email=email,
            api_token=api_token,
            site_subdomain=(
                (site_subdomain or "").strip()
                or str(site.get("site_url") or "")
                .removeprefix("https://")
                .split(".", 1)[0]
            ),
            force=True,
            resolved_site=site,
        )
    except ProviderCredentialError as exc:
        emit_error(
            f"{product_label} credential storage failed", str(exc), verbose=verbose
        )
        raise typer.Exit(code=EXIT_AUTH) from exc
    if result.status != "connected":
        emit_error(
            f"{product_label} authentication failed",
            _auth_failure_message(
                product,
                {
                    "invalid_credentials": AtlassianAuthErrorKind.INVALID_CREDENTIALS,
                    "insufficient_scopes": AtlassianAuthErrorKind.INSUFFICIENT_SCOPES,
                    "product_access_denied": AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED,
                }.get(result.reason or "", last_error),
            ),
            verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH)

    token_storage = integration_token_storage()
    storage_label = (
        "system keychain"
        if token_storage == "keychain"
        else "local credentials file"
    )
    summary = (
        f"Connected {product_label} to {result.site_url}. "
        f"Stored tokens in {storage_label}; metadata saved to {credentials_path()}."
    )
    print_plain_line(
        summary,
        as_json=as_json,
        json_payload={
            "ok": True,
            "provider": product,
            "token_style": result.token_style or "classic",
            "site_url": result.site_url,
            "site_name": result.site_url,
            "cloud_id": result.cloud_id,
            "path": str(credentials_path()),
            "token_storage": token_storage,
            "product_verified": verified_product or product,
        },
    )
