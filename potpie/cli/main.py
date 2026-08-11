"""Host-routed ``potpie`` CLI — the architecture's single spine.

Assembles the per-group command sub-apps (``commands/``) into one Typer app and
binds them to a ``HostShell``. Every command routes
``CLI -> HostShell -> service(s) -> ports``. This is the ``potpie`` console
entrypoint (see ``[project.scripts]``); the in-process ``HostShell`` is the only
composition root for the agent surface.

    Run: ``potpie --help`` (or ``python -m potpie.cli.main --help``)
"""

from __future__ import annotations

import json
import platform
import sys
import traceback
from collections.abc import Sequence
from importlib import metadata
from typing import Final

import typer

from potpie.cli.commands import auth as auth_cmds
from potpie.cli.commands import (
    bootstrap,
    cloud,
    daemon,
    host as host_cmds,
    graph,
    ledger,
    pots,
    telemetry,
)
from potpie.cli.commands import query as query_cmds
from potpie.cli.commands import resource as resource_cmds
from potpie.cli.commands import skills as skills_cmds
from potpie.cli.commands import ui as ui_cmds
from potpie.cli.commands._common import (
    EXIT_VALIDATION,
    argv_requested_json,
    argv_requested_verbose,
    bootstrap_output_flags_from_argv,
    clear_argv_output_flags,
    contract,
    exit_code_for,
    fail,
    is_json,
    is_verbose,
    set_json,
    set_verbose,
)
from potpie.cli.telemetry.context import bind_telemetry_context


def _package_version() -> str:
    try:
        return metadata.version("potpie-context-engine")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _version_callback(value: bool) -> None:
    if not value:
        return
    # `--version` is eager: it fires before the root callback has decided
    # anything, so the only statement about output mode that belongs to *this*
    # command line is the argv scan `run_cli` did on the way in. Consulting it
    # is what stops `potpie --json --version` from answering a machine with two
    # lines of prose — a parse failure the caller only discovers downstream of
    # the pipe, because the exit code says 0.
    if argv_requested_json():
        typer.echo(
            json.dumps(
                {
                    "ok": True,
                    "name": "potpie-context-engine",
                    "version": _package_version(),
                    "python": platform.python_version(),
                    "executable": sys.executable,
                }
            )
        )
    else:
        typer.echo(f"potpie-context-engine {_package_version()}")
        typer.echo(f"python {platform.python_version()} ({sys.executable})")
    raise typer.Exit()


_ROOT_HELP = """\
Potpie context graph CLI (host-routed: CLI → HostShell → services → ports).

First run:
  potpie setup --repo . --agent <harness>
  potpie doctor
  potpie status
"""


def _apply_host_override(raw: str) -> None:
    """Route this one invocation at ``--host``, inside the CLI error contract.

    The validation used to sit in the root callback body — above
    ``configure_error_output`` and outside any ``contract()`` — and both halves
    of that mattered. ``hosts.require_origin`` raises a bare ``ValueError``,
    which :func:`run_cli` does not catch (it handles Click's own exceptions
    only), so ``potpie --json --host bogus host list`` rendered a Rich traceback
    panel on stderr, exited 1, and put nothing at all on stdout: the one output
    shape an agent consumer of ``--json`` cannot parse and cannot branch on.

    The configured-check is here rather than in :func:`hosts.current_origin`
    because ``--host managed`` is *targeting*. Degraded to local it ran against
    the local graph and reported success under the managed label — pot names are
    per-host, so ``default`` answering from the wrong host looks like nothing
    happened. This is the refusal ``potpie host use managed`` already makes, in
    the same envelope, for the same two mistakes.
    """
    from potpie.cli import hosts

    origin = raw.strip().lower()
    # The boundary has to wrap the *reads* as well as the check: resolving the
    # managed endpoint parses the host registry, and an unreadable one raises
    # ContextEngineDisabled, which needs the `unavailable` envelope naming the
    # file rather than a second traceback out of the callback.
    with contract():
        try:
            hosts.require_origin(origin)
        except ValueError as exc:
            # Caught rather than left to `contract()`: its generic ValueError
            # branch has no repair to attach, and naming the way to list the
            # hosts is the entire value of catching a typo here.
            fail(
                code="validation_error",
                message=str(exc),
                next_action="run 'potpie host list'",
                exit_code=EXIT_VALIDATION,
            )
        if origin == hosts.MANAGED and hosts.managed_endpoint() is None:
            fail(
                code="validation_error",
                message="No managed host is configured.",
                next_action="run 'potpie host set <url>'",
                exit_code=EXIT_VALIDATION,
            )
        hosts.set_current_origin(origin)


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
            help="Run this command against a specific host, without changing the active one.",
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
        from potpie_context_engine.bootstrap.runtime_settings import (
            ensure_runtime_environment_loaded,
        )
        from potpie_context_engine.bootstrap import sentry_metrics_runtime

        # `or argv_requested_*`: Click runs this group callback *before* it
        # parses the subcommand's own arguments, so a trailing `potpie pot list
        # --json` arrives here with `json_` false and used to have the argv scan
        # overwritten with that false. The flag is still rejected a moment later
        # (global options are root-only, by design) — but the rejection now goes
        # out in the envelope the caller asked for instead of as bare text.
        set_json(json_ or argv_requested_json())
        set_verbose(verbose or argv_requested_verbose())
        ensure_runtime_environment_loaded()
        configure_error_output(as_json=is_json())
        configure_cli_logging(is_verbose())

        bind_telemetry_context(ctx, json_output=is_json())
        sentry_settings = settings.load_sentry_settings()
        sentry_runtime.configure_cli_sentry(sentry_settings)
        sentry_metrics_runtime.configure_metrics(sentry_settings)
        configure_product_analytics(settings.load_product_analytics_settings())

        # Last, so the refusal it can raise lands on a configured error channel
        # and after `.env` has had its say about POTPIE_MANAGED_URL. Nothing
        # above reads the host registry, so there is nothing to route yet.
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
    app.add_typer(resource_cmds.resource_app, name="resource")
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


#: Options the root callback owns. Deliberately not re-declared per command:
#: two places deciding output mode is what produced the bug below in the first
#: place, and ~100 commands would each be a chance to forget one.
_ROOT_ONLY_FLAGS: Final[frozenset[str]] = frozenset(
    {"--json", "--verbose", "-v", "--version"}
)


def _misplaced_root_flag(exc: Exception) -> str | None:
    """The global flag a caller put *after* the command, if that is the failure.

    Keyed off Click's own ``NoSuchOption`` rather than a scan of argv: Click
    raising it is itself the proof that the command on that path declares no
    such option, so this cannot misfire on the commands that legitimately own a
    flag of their own name (``potpie status --host``, the auth logins).
    """
    from typer._click.exceptions import NoSuchOption

    if not isinstance(exc, NoSuchOption):
        return None
    name = str(getattr(exc, "option_name", "") or "")
    return name if name in _ROOT_ONLY_FLAGS else None


def _reordered_invocation(flag: str, exc: Exception, args: Sequence[str]) -> str:
    """The caller's command with the global flag moved back in front of it.

    Rebuilt from the command names Click resolved, never from the raw argv. The
    first version of this hint echoed every remaining token, which turned
    ``potpie config set <key> <secret> --json`` into a next-action line that
    printed the secret to stdout and into the JSON envelope — reopening, one
    parse-error earlier, the very leak `config set` redacts on its own success
    path. Command names are safe to repeat back; the values typed after them
    are not, so they collapse to an ellipsis the caller can refill themselves.
    """
    ctx = getattr(exc, "ctx", None)
    # Walked off the context chain rather than read off `ctx.command_path`: that
    # string is prefixed with the program name Click took from argv[0], which is
    # one token for the installed `potpie` script but three for `python -m
    # potpie.cli.main`, so slicing it off by count puts the module path into the
    # hint. Every context below the root contributes exactly one command name.
    path: list[str] = []
    node = ctx
    while node is not None and getattr(node, "parent", None) is not None:
        path.append(str(getattr(node, "info_name", "") or ""))
        node = node.parent
    path.reverse()
    if not path:
        return f"put '{flag}' before the command: potpie {flag} <command> ..."
    rest = [a for a in args if a != flag]
    tail = rest[rest.index(path[-1]) + 1 :] if path[-1] in rest else []
    # Click stops resolving at the group that owns the offending flag, so
    # `potpie graph --json catalog` leaves its subcommand unconsumed in argv.
    # Naming it back is safe precisely because it matched a declared command.
    subcommands = getattr(getattr(ctx, "command", None), "commands", {}) or {}
    if tail and tail[0] in subcommands:
        path.append(tail.pop(0))
    return f"re-run as: potpie {flag} {' '.join(path)}{' ...' if tail else ''}"


def run_cli(argv: list[str] | None = None) -> None:
    """Invoke the Typer app with the documented parse-error contract."""
    import click
    from typer._click.exceptions import Abort, ClickException

    from potpie.cli.ui.output import (
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
        flag = _misplaced_root_flag(exc)
        if flag is not None:
            # Routed through `fail` in *both* modes, unlike the usage errors
            # below: this is the one where Click's own text ("No such option:
            # --json") names the symptom and hides the cause — the flag exists,
            # one position to the left.
            fail(
                code="usage_error",
                message=(
                    f"'{flag}' is a global flag and must come before the "
                    f"command: potpie {flag} <command> ..."
                ),
                next_action=_reordered_invocation(flag, exc, args),
            )
        if is_json():
            fail(
                code="usage_error",
                message=_click_error_message(exc),
                next_action="run the command with --help for usage",
            )
        exc.show(file=sys.stderr)
        # Not `exc.exit_code`: Click's UsageError exits 2, which in this CLI's
        # table means "a dependency is unavailable". A typo'd flag and a dead
        # daemon must not be the same number, and the number must not change
        # depending on whether a human or a script is reading — the same missing
        # argument used to exit 2 in prose and 1 as JSON.
        raise typer.Exit(code=exit_code_for("usage_error")) from None
    except (typer.Exit, SystemExit, KeyboardInterrupt):
        # Ordinary control flow: whoever raised these already emitted their
        # contract (or deliberately emitted nothing).
        raise
    except BaseException as exc:  # noqa: BLE001 - last line before the excepthook
        # Typer installs its own `sys.excepthook`, so anything escaping here is
        # rendered as a Rich traceback carrying absolute repo paths, with
        # nothing at all on stdout: the one output shape a `--json` consumer can
        # neither parse nor branch on. Not routed through `contract()` on
        # purpose — that also emits `ce.cli.invocations_total`, which would
        # double-count every command that already went through it.
        from potpie.cli.telemetry.sentry_runtime import capture_unexpected_cli_error

        capture_unexpected_cli_error(
            exc,
            error_code="unexpected_cli_error",
            error_kind="unexpected",
        )
        fail(
            code="unexpected_cli_error",
            message="Unexpected internal error.",
            detail=traceback.format_exc() if is_verbose() else None,
            next_action="re-run with --verbose to see the traceback",
        )
    finally:
        clear_argv_output_flags()

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
