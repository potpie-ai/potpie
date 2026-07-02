"""Service admin command group.

The root daemon currently exposes the active discovery contract through HTTP,
but it does not expose service-admin endpoints yet. Keep these commands as
stable CLI placeholders that return a clear not-implemented error instead of
using the legacy bind-based daemon operation client.
"""

from __future__ import annotations

import typer

from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    fail,
)

service_app = typer.Typer(
    help="Manage the daemon's supporting services.",
    hidden=True,
)


def _service_admin_unavailable() -> None:
    fail(
        code="not_implemented",
        message="service admin commands are not exposed by the HTTP daemon yet",
        next_action="use 'potpie daemon status' for daemon health until admin endpoints are added",
        exit_code=EXIT_UNAVAILABLE,
    )


@service_app.command("up")
def service_up(name: str) -> None:
    with contract():
        del name
        _service_admin_unavailable()


@service_app.command("down")
def service_down(name: str) -> None:
    with contract():
        del name
        _service_admin_unavailable()


@service_app.command("status")
def service_status() -> None:
    with contract():
        _service_admin_unavailable()


@service_app.command("logs")
def service_logs(
    name: str, follow: bool = typer.Option(False, "-f", "--follow")
) -> None:
    with contract():
        del name, follow
        _service_admin_unavailable()


__all__ = ["service_app"]
