"""Host-routed ``potpie`` CLI — the architecture's single spine.

Assembles the per-group command sub-apps (``commands/``) into one Typer app and
binds them to a ``HostShell``. Every command routes
``CLI -> HostShell -> service(s) -> ports``. This is the ``potpie`` console
entrypoint (see ``[project.scripts]``); the in-process ``HostShell`` is the only
composition root for the agent surface.

    Run: ``potpie --help`` (or ``python -m adapters.inbound.cli.host_cli --help``)
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

import typer

from adapters.inbound.cli.commands import auth as auth_cmds
from adapters.inbound.cli.commands import (
    bootstrap,
    cloud,
    daemon,
    graph,
    ledger,
    pots,
    service,
    telemetry,
)
from adapters.inbound.cli.commands import query as query_cmds
from adapters.inbound.cli.commands import skills as skills_cmds
from adapters.inbound.cli.commands import ui as ui_cmds
from adapters.inbound.cli.commands._common import (
    EXIT_VALIDATION,
    bootstrap_output_flags_from_argv,
    fail,
    is_json,
    is_verbose,
    set_json,
    set_verbose,
)
from adapters.inbound.cli.telemetry.context import bind_telemetry_context


def _package_version() -> str:
    try:
        return metadata.version("potpie-context-engine")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _version_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(f"potpie-context-engine {_package_version()}")
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
        from adapters.inbound.cli.telemetry import sentry_runtime, settings
        from adapters.inbound.cli.telemetry.product_analytics import (
            configure_product_analytics,
        )
        from adapters.inbound.cli.ui.output import (
            configure_cli_logging,
            configure_error_output,
        )
        from bootstrap.runtime_settings import ensure_runtime_environment_loaded
        from bootstrap import sentry_metrics_runtime

        set_json(json_)
        set_verbose(verbose)
        ensure_runtime_environment_loaded()
        configure_error_output(as_json=json_)
        configure_cli_logging(verbose)

        bind_telemetry_context(ctx, json_output=json_)
        sentry_settings = settings.load_sentry_settings()
        sentry_runtime.configure_cli_sentry(sentry_settings)
        sentry_metrics_runtime.configure_metrics(sentry_settings)
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


def _click_error_message(exc: Exception) -> str:
    formatter = getattr(exc, "format_message", None)
    if callable(formatter):
        return str(formatter())
    return str(exc)


def run_cli(argv: list[str] | None = None) -> None:
    """Invoke the Typer app with the documented parse-error contract."""
    import click
    from typer._click.exceptions import Abort, ClickException

    from adapters.inbound.cli.ui.output import (
        configure_cli_logging,
        configure_error_output,
    )

    args = list(argv if argv is not None else sys.argv[1:])
    bootstrap_output_flags_from_argv(args)
    if is_json():
        configure_error_output(as_json=True)
    configure_cli_logging(is_verbose())

    try:
        exit_code = app(args, standalone_mode=False)
    except (Abort, click.Abort):
        raise typer.Exit(code=1) from None
    except ClickException as exc:
        if is_json():
            fail(
                code="usage_error",
                message=_click_error_message(exc),
                next_action="run the command with --help for usage",
                exit_code=EXIT_VALIDATION,
            )
        exc.show(file=sys.stderr)
        sys.exit(exc.exit_code)

    if exit_code:
        raise typer.Exit(code=int(exit_code))


def main() -> None:
    try:
        run_cli()
    except typer.Exit as exc:
        # Typer's Exit is not a SystemExit; convert so console-script wrappers
        # exit cleanly without printing exception chains/tracebacks.
        raise SystemExit(exc.exit_code or 0) from None


if __name__ == "__main__":
    main()
