"""Unified Atlassian suite CLI flows."""

from __future__ import annotations

import sys
import time
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from adapters.inbound.cli.auth.atlassian_auth import (
    ProductConnectResult,
    connect_atlassian_product,
    open_url_with_countdown,
)
from adapters.inbound.cli.auth.bitbucket_auth import (
    _bitbucket_failure_message,
    _run_bitbucket_step,
)
from adapters.inbound.cli.commands._common import EXIT_AUTH, get_store
from adapters.inbound.cli.ui.output import emit_error, print_json_blob, print_plain_line
from adapters.inbound.cli.ui.setup_wizard_ui import rich_ui_enabled
from adapters.outbound.cli_auth.env_bootstrap import load_cli_env
from adapters.outbound.cli_auth.provider_config import ATLASSIAN_API_TOKEN_PAGE

_console = Console()


def _print_block(lines: list[str], *, as_json: bool = False) -> None:
    for line in lines:
        print_plain_line(line, as_json=as_json)


def _human() -> bool:
    return rich_ui_enabled(as_json=False)


def _status_mark(status: str) -> tuple[str, str]:
    if status in {"connected", "already_connected"}:
        return "✓", "green"
    if status == "skipped":
        return "–", "yellow"
    return "✗", "red"


def _integration_snapshot(provider: str) -> tuple[str, str | None]:
    """Return (status, site_url) from stored credentials."""
    integration = get_store().get_integration_status(provider)
    if integration.get("authenticated"):
        return "already_connected", integration.get("site_url")
    return "not_connected", None


def _render_status_card(results: dict[str, ProductConnectResult], *, as_json: bool) -> None:
    if as_json or not _human():
        _print_block(
            [
                "Current Atlassian connections:",
                f"  jira       : {_format_site_url(get_store().get_jira_credentials().get('site_url'))}",
                f"  confluence : {_format_site_url(get_store().get_confluence_credentials().get('site_url'))}",
                f"  bitbucket  : {_format_site_url(get_store().get_bitbucket_credentials().get('site_url'))}",
            ]
        )
        return

    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)
    table.add_column(width=12)
    table.add_column()
    for provider in ("jira", "confluence", "bitbucket"):
        result = results.get(provider)
        if result is not None:
            status = result.status
            site_url = result.site_url
        else:
            status, site_url = _integration_snapshot(provider)
        mark, color = _status_mark(status)
        label = Text(provider, style="bold")
        value = Text(
            _format_site_url(site_url)
            if status in {"connected", "already_connected"}
            else "(not connected)",
        )
        table.add_row(Text(mark, style=color), label, value)
    _console.print(
        Panel(
            table,
            title="Current Atlassian connections",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def _format_site_url(site_url: str | None) -> str:
    value = str(site_url or "").strip()
    if not value:
        return "(not connected)"
    return value


def _result_payload(result: ProductConnectResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": result.status}
    if result.site_url:
        payload["site_url"] = result.site_url
    if result.cloud_id:
        payload["cloud_id"] = result.cloud_id
    if result.reason:
        payload["reason"] = result.reason
    return payload


def _render_step_panel(title: str, lines: list[str], *, as_json: bool) -> None:
    if as_json:
        return
    if not _human():
        _print_block([title, *lines])
        return
    body = Text("\n".join(lines))
    _console.print(
        Panel(body, title=title, border_style="yellow", padding=(0, 1))
    )


def _render_result_lines(results: dict[str, ProductConnectResult], *, as_json: bool) -> None:
    if as_json:
        return
    if not _human():
        for provider in ("jira", "confluence", "bitbucket"):
            result = results.get(provider)
            if result is None:
                continue
            mark, _color = _status_mark(result.status)
            suffix = f" ({result.site_url})" if result.site_url else ""
            reason = f" - {result.reason}" if result.reason else ""
            _print_block([f"{mark} {provider}: {result.status}{suffix}{reason}"])
        return

    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)
    table.add_column()
    for provider in ("jira", "confluence", "bitbucket"):
        result = results.get(provider)
        if result is None:
            continue
        mark, color = _status_mark(result.status)
        suffix = f" ({result.site_url})" if result.site_url else ""
        reason = f" - {result.reason}" if result.reason else ""
        table.add_row(
            Text(mark, style=color),
            Text(f"{provider}: {result.status}{suffix}{reason}"),
        )
    _console.print(
        Panel(table, title="Summary", border_style="yellow", padding=(0, 1))
    )


def _render_connection_success(message: str, *, as_json: bool) -> None:
    if as_json:
        return
    if _human():
        _console.print(
            Panel(Text(message, style="green"), border_style="yellow", padding=(0, 1))
        )
        return
    print_plain_line(message, as_json=False)


def _connected_products_message(results: dict[str, ProductConnectResult]) -> str:
    """Build a success message based on which Jira/Confluence products actually connected."""
    connected = [
        p.capitalize()
        for p in ("jira", "confluence")
        if results.get(p) and results[p].status == "connected"
    ]
    if not connected:
        return ""
    return f"Connected {' and '.join(connected)}."


def _print_snapshot() -> None:
    _render_status_card({}, as_json=False)


def _confirm(prompt: str, *, as_json: bool, default: bool = True) -> bool:
    """Ask a single yes/no question (standard CLI style)."""
    if as_json:
        return default
    return typer.confirm(prompt, default=default)


def _prompt_connect_now(prompt: str, *, as_json: bool) -> bool:
    """Backward-compatible alias for tests and callers."""
    return _confirm(prompt, as_json=as_json)


def _render_jira_confluence_panel(*, as_json: bool) -> None:
    _render_step_panel(
        "Jira and Confluence",
        [
            "• One unscoped API token from id.atlassian.com works for both.",
            "• Choose Create API token (without scopes).",
            "• Jira and Confluence may be on different site subdomains — you can update either after the initial attempt.",
        ],
        as_json=as_json,
    )


def _prompt_step1_credentials(
    *,
    email: str | None,
    api_token: str | None,
    site_subdomain: str | None,
) -> tuple[str, str, str]:
    email_value = (email or "").strip() or typer.prompt("Atlassian email").strip()
    site_value = (site_subdomain or "").strip() or typer.prompt(
        "Site subdomain (e.g. myteam for myteam.atlassian.net)"
    ).strip()
    api_token_value = (api_token or "").strip() or typer.prompt(
        "Atlassian API token (id.atlassian.com, without scopes)",
        hide_input=True,
    ).strip()
    return email_value, api_token_value, site_value


def _maybe_connect_product_fallback(
    product: str,
    *,
    email: str,
    api_token: str,
    requested_subdomain: str | None,
) -> ProductConnectResult | None:
    """Try a product on a different subdomain when the caller-supplied flag provides one.

    Used for non-interactive ``--jira-site-subdomain`` / ``--confluence-site-subdomain``.
    """
    if not requested_subdomain:
        return None
    return connect_atlassian_product(
        product,
        email=email,
        api_token=api_token,
        site_subdomain=requested_subdomain,
        force=True,
    )


# Backward-compatible alias kept for any external callers / tests.
def _maybe_connect_confluence_fallback(
    *,
    email: str,
    api_token: str,
    requested_subdomain: str | None,
) -> ProductConnectResult | None:
    return _maybe_connect_product_fallback(
        "confluence", email=email, api_token=api_token, requested_subdomain=requested_subdomain
    )


def _offer_retry_failed_products(
    *,
    results: dict[str, ProductConnectResult],
    email: str,
    api_token: str,
    as_json: bool,
) -> dict[str, ProductConnectResult]:
    """Interactively offer to retry any Jira/Confluence product that failed initial connection.

    Covers Cases B (Jira ok, Confluence not), C (Confluence ok, Jira not), and D (neither ok).
    Reuses the same email and API token — only asks for the missing site subdomain.
    Loops until the user connects successfully or explicitly declines.
    """
    for product in ("jira", "confluence"):
        result = results.get(product)
        if result is None or result.status in {"connected", "already_connected", "skipped"}:
            continue
        name = product.capitalize()
        while True:
            if not typer.confirm(
                f"{name} is not connected. Would you like to connect {name} now?",
                default=True,
            ):
                break
            subdomain = typer.prompt(
                f"{name} site subdomain (e.g. myteam for myteam.atlassian.net)"
            ).strip()
            if not subdomain:
                print_plain_line("Subdomain cannot be empty, please try again.", as_json=False)
                continue
            results[product] = connect_atlassian_product(
                product,
                email=email,
                api_token=api_token,
                site_subdomain=subdomain,
                force=True,
            )
            if results[product].status in {"connected", "already_connected"}:
                break
            # Connection failed again — loop to offer another attempt.
    return results


def _run_step1(
    *,
    force: bool,
    email: str | None,
    api_token: str | None,
    site_subdomain: str | None,
    confluence_site_subdomain: str | None,
    jira_site_subdomain: str | None = None,
    verbose: bool,
    as_json: bool,
) -> dict[str, ProductConnectResult]:
    jira_status = get_store().get_integration_status("jira")
    confluence_status = get_store().get_integration_status("confluence")
    results: dict[str, ProductConnectResult] = {}
    supplied = bool((email or "").strip() and (api_token or "").strip() and (site_subdomain or "").strip())

    if jira_status.get("authenticated") and confluence_status.get("authenticated") and not force:
        results["jira"] = ProductConnectResult(
            product="jira",
            status="already_connected",
            site_url=jira_status.get("site_url"),
            cloud_id=jira_status.get("cloud_id"),
        )
        results["confluence"] = ProductConnectResult(
            product="confluence",
            status="already_connected",
            site_url=confluence_status.get("site_url"),
            cloud_id=confluence_status.get("cloud_id"),
        )
        return results

    if not sys.stdin.isatty() and not force:
        if not supplied:
            emit_error(
                "Atlassian authentication requires a terminal",
                "Run in an interactive shell or pass --email, --api-token, and --site-subdomain.",
            )
            raise typer.Exit(code=1)

    if sys.stdin.isatty():
        if not _prompt_connect_now("Connect Jira and Confluence?", as_json=as_json):
            results["jira"] = ProductConnectResult(product="jira", status="skipped")
            results["confluence"] = ProductConnectResult(product="confluence", status="skipped")
            return results
    else:
        if not as_json:
            print_plain_line("Jira and Confluence", as_json=False, markup=False)

    if not supplied and sys.stdin.isatty():
        if not as_json:
            _render_jira_confluence_panel(as_json=as_json)
        if not as_json and not (api_token or "").strip():
            open_url_with_countdown(
                ATLASSIAN_API_TOKEN_PAGE,
                label="id.atlassian.com",
                timeout_seconds=10,
                open_message="Return here and paste the token when you're ready.",
            )
        email_value, api_token_value, site_value = _prompt_step1_credentials(
            email=email,
            api_token=api_token,
            site_subdomain=site_subdomain,
        )
    elif supplied:
        email_value, api_token_value, site_value = _prompt_step1_credentials(
            email=email,
            api_token=api_token,
            site_subdomain=site_subdomain,
        )
    else:
        results["jira"] = ProductConnectResult(product="jira", status="skipped")
        results["confluence"] = ProductConnectResult(product="confluence", status="skipped")
        return results
    if jira_status.get("authenticated") and not force:
        results["jira"] = ProductConnectResult(
            product="jira",
            status="already_connected",
            site_url=jira_status.get("site_url"),
            cloud_id=jira_status.get("cloud_id"),
        )
    else:
        jira_result = connect_atlassian_product(
            "jira",
            email=email_value,
            api_token=api_token_value,
            site_subdomain=site_value,
            force=force,
        )
        # Non-interactive: if --jira-site-subdomain was provided, try it now.
        if jira_result.status != "connected":
            fallback = _maybe_connect_product_fallback(
                "jira",
                email=email_value,
                api_token=api_token_value,
                requested_subdomain=jira_site_subdomain,
            )
            if fallback is not None:
                jira_result = fallback
        results["jira"] = jira_result

    if confluence_status.get("authenticated") and not force:
        results["confluence"] = ProductConnectResult(
            product="confluence",
            status="already_connected",
            site_url=confluence_status.get("site_url"),
            cloud_id=confluence_status.get("cloud_id"),
        )
    else:
        confluence_result = connect_atlassian_product(
            "confluence",
            email=email_value,
            api_token=api_token_value,
            site_subdomain=site_value,
            force=force,
        )
        # Non-interactive: if --confluence-site-subdomain was provided, try it now.
        if confluence_result.status != "connected":
            fallback = _maybe_connect_product_fallback(
                "confluence",
                email=email_value,
                api_token=api_token_value,
                requested_subdomain=confluence_site_subdomain,
            )
            if fallback is not None:
                confluence_result = fallback
        results["confluence"] = confluence_result

    # Interactive: offer to retry any product that still failed (Cases B / C / D).
    if sys.stdin.isatty():
        results = _offer_retry_failed_products(
            results=results,
            email=email_value,
            api_token=api_token_value,
            as_json=as_json,
        )

    # Show summary whenever at least one product was attempted (not skipped),
    # so partial failures are never silent.
    attempted = any(r.status != "skipped" for r in results.values())
    if attempted:
        _render_result_lines(results, as_json=as_json)
        msg = _connected_products_message(results)
        if msg:
            _render_connection_success(msg, as_json=as_json)
            if _human():
                time.sleep(1)
    return results


def run_atlassian_suite_login(
    *,
    force: bool = False,
    as_json: bool = False,
    verbose: bool = False,
    skip_bitbucket: bool = False,
    email: str | None = None,
    api_token: str | None = None,
    site_subdomain: str | None = None,
    confluence_site_subdomain: str | None = None,
    jira_site_subdomain: str | None = None,
    bitbucket_api_token: str | None = None,
) -> None:
    load_cli_env()
    if not as_json:
        _print_snapshot()

    step1 = _run_step1(
        force=force,
        email=email,
        api_token=api_token,
        site_subdomain=site_subdomain,
        confluence_site_subdomain=confluence_site_subdomain,
        jira_site_subdomain=jira_site_subdomain,
        verbose=verbose,
        as_json=as_json,
    )
    resolved_email = (email or "").strip()
    if not resolved_email:
        for provider in ("jira", "confluence"):
            status = get_store().get_integration_status(provider)
            resolved_email = str(status.get("email") or "").strip()
            if resolved_email:
                break
    bitbucket = _run_bitbucket_step(
        force=force,
        skip_bitbucket=skip_bitbucket,
        bitbucket_api_token=bitbucket_api_token,
        email=resolved_email,
        as_json=as_json,
    )

    products = {
        "jira": _result_payload(step1["jira"]),
        "confluence": _result_payload(step1["confluence"]),
        "bitbucket": _result_payload(bitbucket),
    }
    ok = any(
        result["status"] in {"connected", "already_connected"}
        for result in products.values()
    )
    payload = {"ok": ok, "products": products}

    # Always surface a Bitbucket error — even when Jira/Confluence are already
    # connected (ok=True), a failed Bitbucket attempt must not be silently ignored.
    if bitbucket.status == "not_connected":
        emit_error(
            "Bitbucket authentication failed",
            _bitbucket_failure_message(bitbucket.reason),
        )

    if not ok and any(
        result["status"] != "skipped" for result in products.values()
    ):
        raise typer.Exit(code=EXIT_AUTH)
    if as_json:
        print_json_blob(payload, as_json=True)
        return


def run_atlassian_suite_logout(*, as_json: bool = False) -> None:
    load_cli_env()
    get_store().clear_atlassian_suite_credentials()
    print_plain_line(
        "Cleared Jira, Confluence, Bitbucket, and legacy Atlassian credentials.",
        as_json=as_json,
        json_payload={"ok": True},
    )


def build_atlassian_suite_status() -> list[dict[str, Any]]:
    load_cli_env()
    return [
        get_store().get_integration_status("jira"),
        get_store().get_integration_status("confluence"),
        get_store().get_integration_status("bitbucket"),
    ]
