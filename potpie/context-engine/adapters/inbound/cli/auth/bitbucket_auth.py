"""Standalone Bitbucket Cloud CLI authentication flows."""

from __future__ import annotations

import select
import sys
import webbrowser
from dataclasses import dataclass
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from adapters.inbound.cli.commands._common import EXIT_AUTH, get_store
from adapters.inbound.cli.ui.output import emit_error, print_plain_line
from adapters.inbound.cli.ui.setup_wizard_ui import rich_ui_enabled
from adapters.outbound.cli_auth.bitbucket_client import verify_bitbucket_api_token
from adapters.outbound.cli_auth.credentials_store import (
    ProviderCredentialError,
    credentials_path,
    integration_token_storage,
)
from adapters.outbound.cli_auth.env_bootstrap import load_cli_env
from adapters.outbound.cli_auth.provider_config import BITBUCKET_API_TOKEN_PAGE

_console = Console()


@dataclass(frozen=True)
class ProductConnectResult:
    product: str
    status: str
    site_url: str | None = None
    cloud_id: str | None = None
    reason: str | None = None
    token_style: str | None = None


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


def _confirm(prompt: str, *, as_json: bool, default: bool = True) -> bool:
    if as_json:
        return default
    return typer.confirm(prompt, default=default)


def _prompt_connect_now(prompt: str, *, as_json: bool) -> bool:
    return _confirm(prompt, as_json=as_json)


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


def bitbucket_failure_message(reason: str | None) -> str:
    if reason == "invalid_credentials":
        return "Bitbucket rejected the email or API token."
    if reason == "insufficient_scopes":
        return (
            "Bitbucket token is missing required read scopes: "
            "read:user:bitbucket, read:workspace:bitbucket, "
            "read:repository:bitbucket, read:pullrequest:bitbucket."
        )
    return "Bitbucket verification failed."


def _bitbucket_failure_message(reason: str | None) -> str:
    """Backward-compatible alias for suite callers and tests."""
    return bitbucket_failure_message(reason)


def run_bitbucket_step(
    *,
    force: bool,
    skip_bitbucket: bool,
    bitbucket_api_token: str | None,
    email: str | None,
    as_json: bool,
) -> ProductConnectResult:
    if skip_bitbucket:
        return ProductConnectResult(product="bitbucket", status="skipped")
    status = get_store().get_integration_status("bitbucket")
    if status.get("authenticated") and not force:
        return ProductConnectResult(
            product="bitbucket",
            status="already_connected",
        )
    if not bitbucket_api_token:
        if not sys.stdin.isatty():
            return ProductConnectResult(product="bitbucket", status="skipped")
        if not _prompt_connect_now("Connect Bitbucket?", as_json=as_json):
            return ProductConnectResult(product="bitbucket", status="skipped")
        if not as_json:
            _render_step_panel(
                "Bitbucket",
                [
                    "• Create API token with scopes at id.atlassian.com.",
                    "• Select Bitbucket as the app.",
                    "• Required read scopes:",
                    "  - read:user:bitbucket",
                    "  - read:workspace:bitbucket",
                    "  - read:pullrequest:bitbucket",
                    "  - read:repository:bitbucket",
                    "• Do not select write or admin scopes.",
                ],
                as_json=as_json,
            )
            open_url_with_countdown(
                BITBUCKET_API_TOKEN_PAGE,
                label="id.atlassian.com",
                timeout_seconds=10,
                open_message="Return here and paste the token when you're ready.",
            )
        email_value = typer.prompt("Bitbucket email").strip()
        bitbucket_api_token = typer.prompt(
            "Bitbucket API token",
            hide_input=True,
        ).strip()
    else:
        email_value = (email or status.get("email") or "").strip()
        if not email_value:
            if not sys.stdin.isatty():
                return ProductConnectResult(
                    product="bitbucket",
                    status="not_connected",
                    reason="invalid_credentials",
                )
            email_value = typer.prompt("Bitbucket email").strip()

    result = verify_bitbucket_api_token(email_value, bitbucket_api_token)
    if not result.ok:
        return ProductConnectResult(
            product="bitbucket",
            status="not_connected",
            reason=result.error_kind,
        )
    try:
        get_store().save_bitbucket_credentials(
            {
                "email": email_value,
                "api_token": bitbucket_api_token,
                "account_name": result.display_name,
            }
        )
    except ProviderCredentialError as exc:
        emit_error("Bitbucket credential storage failed", str(exc))
        raise typer.Exit(code=EXIT_AUTH) from exc
    connected = ProductConnectResult(product="bitbucket", status="connected")
    _render_result_lines({"bitbucket": connected}, as_json=as_json)
    _render_connection_success("Connected Bitbucket.", as_json=as_json)
    return connected


def _run_bitbucket_step(
    *,
    force: bool,
    skip_bitbucket: bool,
    bitbucket_api_token: str | None,
    email: str | None,
    as_json: bool,
) -> ProductConnectResult:
    """Backward-compatible alias for suite callers and tests."""
    return run_bitbucket_step(
        force=force,
        skip_bitbucket=skip_bitbucket,
        bitbucket_api_token=bitbucket_api_token,
        email=email,
        as_json=as_json,
    )


def run_bitbucket_login(
    *,
    force: bool = False,
    as_json: bool = False,
    verbose: bool = False,
    email: str | None = None,
    bitbucket_api_token: str | None = None,
) -> None:
    """Connect Bitbucket Cloud with a scoped API token (standalone login)."""
    load_cli_env()
    try:
        status = get_store().get_integration_status("bitbucket")
    except ProviderCredentialError as exc:
        emit_error("Bitbucket credential lookup failed", str(exc), verbose=verbose)
        status = {}

    if status.get("authenticated") and not force:
        print_plain_line(
            "Bitbucket is already connected.",
            as_json=as_json,
            json_payload={
                "ok": True,
                "already_connected": True,
                "provider": "bitbucket",
                "email": status.get("email"),
                "login": status.get("login"),
                "site_url": status.get("site_url"),
            },
        )
        return

    resolved_email = (email or "").strip()
    if not resolved_email:
        for provider in ("bitbucket", "jira", "confluence"):
            provider_status = get_store().get_integration_status(provider)
            resolved_email = str(provider_status.get("email") or "").strip()
            if resolved_email:
                break

    result = run_bitbucket_step(
        force=force,
        skip_bitbucket=False,
        bitbucket_api_token=bitbucket_api_token,
        email=resolved_email or None,
        as_json=as_json,
    )

    if result.status == "connected":
        token_storage = integration_token_storage()
        storage_label = (
            "system keychain"
            if token_storage == "keychain"
            else "local credentials file"
        )
        print_plain_line(
            f"Connected Bitbucket. Stored tokens in {storage_label}; "
            f"metadata saved to {credentials_path()}.",
            as_json=as_json,
            json_payload={
                "ok": True,
                "provider": "bitbucket",
                "email": resolved_email,
                "path": str(credentials_path()),
                "token_storage": token_storage,
            },
        )
        return

    if result.status == "not_connected":
        emit_error(
            "Bitbucket authentication failed",
            bitbucket_failure_message(result.reason),
            verbose=verbose,
        )
        raise typer.Exit(code=EXIT_AUTH)
