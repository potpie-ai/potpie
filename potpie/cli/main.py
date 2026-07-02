"""Host-routed ``potpie`` CLI — the architecture's single spine.

Assembles the per-group command sub-apps (``commands/``) into one Typer app and
binds them to a ``HostShell``. Every command routes
``CLI -> HostShell -> service(s) -> ports``. This is the ``potpie`` console
entrypoint (see ``[project.scripts]``); the in-process ``HostShell`` is the only
composition root for the agent surface.

    Run: ``potpie --help`` (or ``python -m potpie.cli.main --help``)
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

import typer

from potpie.cli.commands import auth as auth_cmds
from potpie.cli.commands import (
    bootstrap,
    cloud,
    daemon,
    graph,
    ledger,
    pots,
    service,
    telemetry,
)
from potpie.cli.commands import query as query_cmds
from potpie.cli.commands import skills as skills_cmds
from potpie.cli.commands import ui as ui_cmds
from potpie.cli.commands._common import set_json, set_verbose
from potpie.cli.telemetry.context import bind_telemetry_context


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _version_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(f"potpie {_distribution_version('potpie')}")
    typer.echo(
        f"potpie-context-engine {_distribution_version('potpie-context-engine')}"
    )
    typer.echo(f"python {platform.python_version()} ({sys.executable})")
    raise typer.Exit()


def build_app() -> typer.Typer:
    app = typer.Typer(
        name="potpie",
        help="Potpie context graph CLI (host-routed: CLI → HostShell → services → ports).",
        no_args_is_help=True,
        add_completion=False,
    )

    @app.callback()
    def _root(
        ctx: typer.Context,
        json_: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
        verbose: bool = typer.Option(
            False, "--verbose", "-v", help="Verbose tracebacks on errors."
        ),
        version: bool = typer.Option(
            False,
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version information and exit.",
        ),
    ) -> None:
        from potpie.cli.telemetry import sentry_runtime, settings
        from potpie.cli.telemetry.product_analytics import (
            configure_product_analytics,
        )
        from potpie.cli.ui.output import (
            configure_cli_logging,
            configure_error_output,
        )
        from potpie.runtime.settings import ensure_runtime_environment_loaded

        set_json(json_)
        set_verbose(verbose)
        ensure_runtime_environment_loaded()
        configure_error_output(as_json=json_)
        configure_cli_logging(verbose)

        bind_telemetry_context(ctx, json_output=json_)
        sentry_settings = settings.load_sentry_settings()
        sentry_runtime.configure_cli_sentry(sentry_settings)
        configure_product_analytics(settings.load_product_analytics_settings())

    # Top-level commands (the four-tool surface + bootstrap + auth/login).
    query_cmds.register(app)
    bootstrap.register(app)
    auth_cmds.register(app)
    ui_cmds.register(app)

    # Command groups (one per cli-flow.md section).
    app.add_typer(pots.pot_app, name="pot")
    app.add_typer(pots.source_app, name="source")
    app.add_typer(daemon.daemon_app, name="daemon")
    app.add_typer(service.service_app, name="service")
    app.add_typer(ledger.ledger_app, name="ledger")
    app.add_typer(graph.graph_app, name="graph")
    app.add_typer(graph.timeline_app, name="timeline")
    app.add_typer(graph.backend_app, name="backend")
    app.add_typer(skills_cmds.skills_app, name="skills")
    app.add_typer(cloud.cloud_app, name="cloud")
    app.add_typer(telemetry.telemetry_app, name="telemetry")

    return app


app = build_app()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
