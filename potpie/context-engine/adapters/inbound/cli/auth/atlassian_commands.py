"""CLI commands for the unified Atlassian suite integration."""

from __future__ import annotations

import typer

from adapters.inbound.cli.auth.atlassian_suite_auth import (
    build_atlassian_suite_status,
    run_atlassian_suite_login,
    run_atlassian_suite_logout,
)
from adapters.inbound.cli.ui.output import print_json_blob, print_plain_line
from adapters.inbound.cli.commands._common import is_json, is_verbose

atlassian_app = typer.Typer(help="Atlassian suite integration.")


def _flags() -> tuple[bool, bool]:
    return is_json(), is_verbose()


@atlassian_app.command("login")
def atlassian_login(
    force: bool = typer.Option(False, "--force"),
    skip_bitbucket: bool = typer.Option(False, "--skip-bitbucket"),
    email: str | None = typer.Option(None, "--email"),
    api_token: str | None = typer.Option(None, "--api-token"),
    site_subdomain: str | None = typer.Option(None, "--site-subdomain"),
    confluence_site_subdomain: str | None = typer.Option(
        None, "--confluence-site-subdomain"
    ),
    bitbucket_api_token: str | None = typer.Option(None, "--bitbucket-api-token"),
) -> None:
    j, v = _flags()
    run_atlassian_suite_login(
        force=force,
        as_json=j,
        verbose=v,
        skip_bitbucket=skip_bitbucket,
        email=email,
        api_token=api_token,
        site_subdomain=site_subdomain,
        confluence_site_subdomain=confluence_site_subdomain,
        bitbucket_api_token=bitbucket_api_token,
    )


@atlassian_app.command("logout")
def atlassian_logout() -> None:
    j, _ = _flags()
    run_atlassian_suite_logout(as_json=j)


@atlassian_app.command("status")
def atlassian_status() -> None:
    j, _ = _flags()
    rows = build_atlassian_suite_status()
    if j:
        print_json_blob({"integrations": rows}, as_json=True)
        return
    for row in rows:
        provider = row["provider"]
        if not row.get("authenticated"):
            print_plain_line(f"{provider}: not authenticated", as_json=False)
            continue
        parts = [f"{provider}: authenticated"]
        if row.get("email"):
            parts.append(f"email={row['email']}")
        if row.get("login"):
            parts.append(f"login={row['login']}")
        if row.get("site_url"):
            parts.append(f"url={row['site_url']}")
        print_plain_line("  ".join(parts), as_json=False, markup=False)
