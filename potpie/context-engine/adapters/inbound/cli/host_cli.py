"""Host-routed ``potpie`` CLI — the architecture's single spine.

Assembles the per-group command sub-apps (``commands/``) into one Typer app and
binds them to a ``HostShell``. Every command routes
``CLI -> HostShell -> service(s) -> ports``. This is the ``potpie`` console
entrypoint (see ``[project.scripts]``); the in-process ``HostShell`` is the only
composition root for the agent surface.

    Run: ``potpie --help`` (or ``python -m adapters.inbound.cli.host_cli --help``)
"""

from __future__ import annotations

import typer

from adapters.inbound.cli.commands import auth as auth_cmds
from adapters.inbound.cli.commands import bootstrap, cloud, daemon, graph, ledger, pots
from adapters.inbound.cli.commands import ingest as ingest_cmds
from adapters.inbound.cli.commands import query as query_cmds
from adapters.inbound.cli.commands import skills as skills_cmds
from adapters.inbound.cli.commands._common import set_json, set_verbose


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
    ) -> None:
        from adapters.outbound.cli_auth.env_bootstrap import load_cli_env
        from adapters.inbound.cli.ui.output import (
            configure_cli_logging,
            configure_error_output,
        )

        set_json(json_)
        set_verbose(verbose)
        configure_error_output(as_json=json_)
        configure_cli_logging(verbose)
        load_cli_env()
        from adapters.inbound.cli.telemetry.context import bind_telemetry_context
        from adapters.inbound.cli.telemetry.sentry_runtime import configure_cli_sentry
        from adapters.inbound.cli.telemetry.settings import load_sentry_settings
        from bootstrap.sentry_metrics_runtime import configure_metrics

        bind_telemetry_context(ctx, json_output=json_)
        settings = load_sentry_settings()
        configure_cli_sentry(settings)
        configure_metrics(settings)

    # Top-level commands (the four-tool surface + bootstrap + auth/login).
    query_cmds.register(app)
    bootstrap.register(app)
    auth_cmds.register(app)

    # Command groups (one per cli-flow.md section).
    app.add_typer(pots.pot_app, name="pot")
    app.add_typer(pots.source_app, name="source")
    app.add_typer(daemon.daemon_app, name="daemon")
    app.add_typer(ingest_cmds.ingest_app, name="ingest")
    app.add_typer(ledger.ledger_app, name="ledger")
    app.add_typer(graph.graph_app, name="graph")
    app.add_typer(graph.backend_app, name="backend")
    app.add_typer(skills_cmds.skills_app, name="skills")
    app.add_typer(cloud.cloud_app, name="cloud")

    return app


app = build_app()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
