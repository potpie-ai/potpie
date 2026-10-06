"""Typed-client ``potpie`` CLI — the architecture's product entrypoint.

Assembles the per-group command sub-apps (``commands/``) into one Typer app and
routes context operations through ``EngineClient`` and Potpie-owned operations
through finite root services. This is the ``potpie`` console entrypoint (see
``[project.scripts]``). ``CONTEXT_ENGINE_HOST_MODE`` selects the local or
canonical daemon ``EngineClient`` without changing command contracts.

    Run: ``potpie --help`` (or ``python -m potpie.cli.main --help``)
"""

from __future__ import annotations

import json
import platform
import sys

import click
import typer

from potpie import build_info
from potpie.cli.commands import auth as auth_cmds
from potpie.cli.commands import (
    bootstrap,
    cloud,
    daemon,
    graph,
    ledger,
    pots,
    telemetry,
)
from potpie.cli.commands import query as query_cmds
from potpie.cli.commands import skills as skills_cmds
from potpie.cli.commands import ui as ui_cmds
from potpie.cli.commands._common import (
    EXIT_VALIDATION,
    bootstrap_output_flags_from_argv,
    fail,
    is_json,
    is_verbose,
    set_json,
    set_verbose,
)
from potpie.cli.telemetry.context import bind_telemetry_context


def _version_callback(value: bool) -> None:
    if not value:
        return
    # `potpie` is the distribution that owns this command; the engine is the
    # library under it. Both versions move only at releases, so the build rev
    # is what identifies the code (see potpie.build_info).
    info = build_info.describe()
    # `--version` is eager and fires before the root callback applies
    # `--json`; `run_cli` has already applied it from argv by then.
    if is_json():
        typer.echo(
            json.dumps(
                {
                    **info,
                    "python": platform.python_version(),
                    "executable": sys.executable,
                }
            )
        )
    else:
        typer.echo(build_info.human_line(info))
        typer.echo(f"{info['engine']['name']} {info['engine']['version']}")
        typer.echo(f"python {platform.python_version()} ({sys.executable})")
    raise typer.Exit()


_ROOT_HELP = """\
Potpie context graph CLI (typed context clients and Potpie-owned services).

First run:
  potpie setup --repo . --agent <harness>
  potpie doctor
  potpie status
"""


def build_app() -> typer.Typer:
    app = typer.Typer(
        name="potpie",
        help=_ROOT_HELP,
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
        from potpie.runtime.settings import (
            ensure_runtime_environment_loaded,
        )

        set_json(json_)
        set_verbose(verbose)
        ensure_runtime_environment_loaded()
        configure_error_output(as_json=json_)
        configure_cli_logging(verbose)

        bind_telemetry_context(ctx, json_output=json_)
        # Arms crash capture and routes metrics to the telemetry spool; the
        # Sentry SDK is initialised only if an unexpected error needs reporting.
        sentry_runtime.configure_cli_sentry(settings.load_sentry_settings())
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
    app.add_typer(ledger.ledger_app, name="ledger")
    app.add_typer(graph.graph_app, name="graph")
    app.add_typer(graph.timeline_app, name="timeline")
    app.add_typer(graph.backend_app, name="backend")
    app.add_typer(skills_cmds.skills_app, name="skills")
    # Keep cloud discoverable but below the local happy path — managed routing
    # is still in development (see cli-flow.md).
    app.add_typer(
        cloud.cloud_app,
        name="cloud",
        rich_help_panel="Coming soon",
    )
    app.add_typer(telemetry.telemetry_app, name="telemetry")

    return app


app = build_app()


def _click_error_message(exc: Exception) -> str:
    formatter = getattr(exc, "format_message", None)
    if callable(formatter):
        return str(formatter())
    return str(exc)


def _is_click_exception(exc: Exception) -> bool:
    """Recognize public Click errors and Typer's vendored Click errors."""
    typer_exception = getattr(typer, "TyperException", None)
    is_typer_exception = isinstance(typer_exception, type) and isinstance(
        exc, typer_exception
    )
    is_legacy_typer_click = exc.__class__.__module__ == "typer._click.exceptions"
    return (
        isinstance(exc, click.ClickException)
        or is_typer_exception
        or is_legacy_typer_click
    ) and callable(getattr(exc, "show", None))


def run_cli(argv: list[str] | None = None) -> None:
    """Invoke the Typer app with the documented parse-error contract."""
    from potpie.cli.ui.output import (
        configure_cli_logging,
        configure_error_output,
        configure_output_streams,
    )

    # Help is eager: the root callback runs too late to protect its rendering.
    configure_output_streams()
    args = list(argv if argv is not None else sys.argv[1:])
    bootstrap_output_flags_from_argv(args)
    if is_json():
        configure_error_output(as_json=True)
    configure_cli_logging(is_verbose())

    try:
        exit_code = app(args, standalone_mode=False)
    except (typer.Abort, click.Abort):
        raise typer.Exit(code=1) from None
    except Exception as exc:
        if not _is_click_exception(exc):
            raise
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
