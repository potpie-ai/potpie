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
from collections.abc import Sequence

import click
import typer

from potpie import build_info
from potpie.cli.commands import auth as auth_cmds
from potpie.cli.commands import (
    bootstrap,
    cloud,
    daemon,
    document,
    graph,
    ledger,
    pots,
    telemetry,
)
from potpie.cli.commands import host as host_cmds
from potpie.cli.commands import query as query_cmds
from potpie.cli.commands import resource as resource_cmds
from potpie.cli.commands import skills as skills_cmds
from potpie.cli.commands import ui as ui_cmds
from potpie.cli.commands._common import (
    EXIT_VALIDATION,
    bootstrap_output_flags_from_argv,
    contract,
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


def _apply_host_override(raw: str) -> None:
    """Route this one invocation at ``--host``, inside the CLI error contract.

    ``--host managed`` is targeting, so an unconfigured or unusable managed
    host is refused here rather than degraded to local: pot names are per-host
    labels, and ``default`` answering from the wrong graph looks like success.
    The text comes from the registry because only one of the two ways to have
    no usable managed host is fixed by ``host set``.
    """
    from potpie.config import host_registry

    origin = raw.strip().lower()
    # The boundary wraps the reads too: resolving the managed host parses the
    # registry, and an unreadable one must surface as the `unavailable`
    # envelope naming the file, not as a traceback out of the callback.
    with contract():
        try:
            host_registry.require_origin(origin)
        except ValueError as exc:
            fail(
                code="validation_error",
                message=str(exc),
                next_action="run 'potpie host list'",
                exit_code=EXIT_VALIDATION,
            )
        if origin == host_registry.MANAGED and host_registry.managed_endpoint() is None:
            message, next_action = host_registry.managed_unconfigured_refusal()
            fail(
                code="validation_error",
                message=message,
                next_action=next_action,
                exit_code=EXIT_VALIDATION,
            )
        host_registry.set_current_origin(origin)


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
        host: str = typer.Option(
            "",
            "--host",
            metavar="local|managed",
            help=(
                "Run this command against a specific host, without changing "
                "the active one."
            ),
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
        from potpie.config import host_registry
        from potpie.runtime.settings import (
            ensure_runtime_environment_loaded,
        )

        # The override is per invocation; an in-process caller that runs the
        # app twice must not inherit the previous run's `--host`.
        host_registry.set_current_origin(None)
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

        # Last, so a refusal lands on a configured error channel and after
        # `.env` has had its say about POTPIE_MANAGED_URL.
        if host:
            _apply_host_override(host)

    # Top-level commands (the four-tool surface + bootstrap + auth/login).
    query_cmds.register(app)
    bootstrap.register(app)
    auth_cmds.register(app)
    ui_cmds.register(app)

    # Command groups (one per cli-flow.md section).
    app.add_typer(host_cmds.host_app, name="host")
    app.add_typer(pots.pot_app, name="pot")
    app.add_typer(pots.source_app, name="source")
    app.add_typer(daemon.daemon_app, name="daemon")
    app.add_typer(ledger.ledger_app, name="ledger")
    app.add_typer(graph.graph_app, name="graph")
    app.add_typer(graph.timeline_app, name="timeline")
    app.add_typer(graph.backend_app, name="backend")
    app.add_typer(skills_cmds.skills_app, name="skills")
    app.add_typer(resource_cmds.resource_app, name="resource")
    app.add_typer(document.document_app, name="document")
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


#: Root options a caller may also type after the command. Click raises
#: ``NoSuchOption`` only for a command that declares no option of that name, so
#: a command that owns its own ``--host`` (``gitbucket login --host <url>``)
#: keeps it.
_TRAILING_ROOT_OPTIONS: frozenset[str] = frozenset({"--host"})


def _misplaced_root_option(exc: Exception) -> str | None:
    """The root option typed after a command that does not declare it."""
    if type(exc).__name__ != "NoSuchOption":
        return None
    name = str(getattr(exc, "option_name", "") or "")
    return name if name in _TRAILING_ROOT_OPTIONS else None


def _hoist_root_option(option: str, args: Sequence[str]) -> list[str] | None:
    """``args`` with ``option`` and its value moved in front of the command.

    Only the tokens before a ``--`` separator are the caller's options. ``None``
    when there is nothing safe to move: the option is typed more than once, or
    it has no value after it.
    """
    rest = list(args)
    end = rest.index("--") if "--" in rest else len(rest)
    hits = [
        index
        for index in range(end)
        if rest[index] == option or rest[index].startswith(f"{option}=")
    ]
    if len(hits) != 1:
        return None
    index = hits[0]
    if rest[index] != option:
        token = rest.pop(index)
        return [token, *rest]
    if index + 1 >= end or rest[index + 1].startswith("-"):
        return None
    value = rest[index + 1]
    del rest[index : index + 2]
    return [option, value, *rest]


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
        try:
            exit_code = app(args, standalone_mode=False)
        except Exception as exc:
            option = _misplaced_root_option(exc)
            hoisted = _hoist_root_option(option, args) if option else None
            if hoisted is None:
                raise
            # Click refuses while parsing the command's options, before its
            # body runs or anything is emitted, so `status --host managed` is
            # re-run as `--host managed status`: the same command line, spelled
            # the one way the parser accepts.
            exit_code = app(hoisted, standalone_mode=False)
    except (typer.Abort, click.Abort):
        raise typer.Exit(code=1) from None
    except Exception as exc:
        if not _is_click_exception(exc):
            raise
        option = _misplaced_root_option(exc)
        if option is not None:
            # Click's "No such option: --host" hides that the option exists one
            # position to the left.
            fail(
                code="usage_error",
                message=(
                    f"'{option}' is a global option and takes one value: "
                    f"potpie {option} <local|managed> <command> ..."
                ),
                next_action=f"put '{option} <local|managed>' before the command",
                exit_code=EXIT_VALIDATION,
            )
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
