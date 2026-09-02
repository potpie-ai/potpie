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
from typing import Final

import typer

from potpie import build_info
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


def _version_callback(value: bool) -> None:
    if not value:
        return
    # The distribution that owns this command is `potpie`; the engine is what
    # it is built on. Reporting the engine alone under its own name -- which is
    # what this did -- told a reader the version of a library and nothing about
    # the CLI they had run, and both numbers are constants anyway: the rev in
    # `build` is the part that identifies the code (see potpie.build_info).
    info = build_info.describe()
    # `--version` is eager: it fires before the root callback has decided
    # anything, so the only statement about output mode that belongs to *this*
    # command line is the argv scan `run_cli` did on the way in. Consulting it
    # is what stops `potpie --json --version` from answering a machine with two
    # lines of prose — a parse failure the caller only discovers downstream of
    # the pipe, because the exit code says 0.
    #
    # `is_json()` first, because the scan is retired the moment the root
    # callback applies it (see `set_json`), and `potpie pot list --version
    # --json` reaches this eager callback on the *second* parse — after the
    # first one already consumed the scan. The applied mode is the one that
    # survives a re-parse; the scan is what covers a command line that never
    # got that far.
    if is_json() or argv_requested_json():
        typer.echo(
            json.dumps(
                {
                    "ok": True,
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
            # The text comes from the registry, not from here: there are two
            # ways to have no usable managed host and only one of them is fixed
            # by `host set`. A `POTPIE_MANAGED_URL` that is set but unusable
            # reads as *absent* to every router by design, and answering it with
            # "run 'potpie host set <url>'" tells the caller to write a file
            # their own environment override makes irrelevant.
            message, next_action = hosts.managed_unconfigured_refusal()
            fail(
                code="validation_error",
                message=message,
                next_action=next_action,
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
        sentry_metrics_runtime.configure_metrics(
            sentry_settings, short_lived_process=True
        )
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
#: place, and ~100 commands would each be a chance to forget one. Being declared
#: in one place is not the same as being *typeable* in one place — see
#: :func:`_hoist_root_flag`.
_ROOT_ONLY_FLAGS: Final[frozenset[str]] = frozenset(
    {"--json", "--verbose", "-v", "--version", "--host"}
)

#: The root flags that consume the token after them. Hoisting one has to carry
#: its value along or ``potpie status --host managed`` becomes ``potpie --host
#: status`` — a different, and silently wrong, command line.
_ROOT_VALUE_FLAGS: Final[frozenset[str]] = frozenset({"--host"})

#: How a value-taking root flag is spelled back to the caller. Only the
#: invocations that supplied *no* value reach the hint (the rest are hoisted
#: value and all), so repeating the bare flag would echo the mistake instead of
#: naming the fix.
_ROOT_FLAG_SPELLING: Final[dict[str, str]] = {"--host": "--host <local|managed>"}


def _misplaced_root_flag(exc: Exception) -> str | None:
    """The global flag a caller put *after* the command, if that is the failure.

    Keyed off Click's own ``NoSuchOption`` rather than a scan of argv: Click
    raising it is itself the proof that the command on that path declares no
    such option, so this cannot misfire on the commands that legitimately own a
    flag of their own name (``potpie gitbucket login --host <url>``).
    """
    from typer._click.exceptions import NoSuchOption

    if not isinstance(exc, NoSuchOption):
        return None
    name = str(getattr(exc, "option_name", "") or "")
    return name if name in _ROOT_ONLY_FLAGS else None


def _hoist_root_flag(flag: str, args: Sequence[str]) -> list[str] | None:
    """``args`` with a misplaced global ``flag`` moved in front of the command.

    Root-only is a statement about where the flag is *declared*, not about where
    a caller may type it. One accepted position is a coin flip every agent
    consumer has to win on the first try, and the losing side got Click's prose
    on stderr from a command line that had just said it could only read JSON.
    Click has already raised ``NoSuchOption``, which is proof the command
    declares no option of this name, so moving the token to the front cannot
    change what the command means — it can only make it parse.

    Only the region before a ``--`` separator is touched: everything after it is
    the caller's data, and a token there never produced the ``NoSuchOption``
    that got us here in the first place.

    ``None`` when there is nothing safe to move — the flag is not in argv, or it
    takes a value and the next token is another option (or missing). Click's own
    "requires an argument" is the better error for that, and the refusal in
    :func:`run_cli` still covers it.
    """
    rest = list(args)
    end = rest.index("--") if "--" in rest else len(rest)
    for index in range(end):
        token = rest[index]
        if token.startswith(f"{flag}="):
            del rest[index]
            return [token, *rest]
        if token != flag:
            continue
        del rest[index]
        if flag not in _ROOT_VALUE_FLAGS:
            return [flag, *rest]
        if index >= len(rest) or rest[index].startswith("-"):
            return None
        value = rest.pop(index)
        return [flag, value, *rest]
    return None


def _reordered_invocation(flag: str, exc: Exception, args: Sequence[str]) -> str:
    """The caller's command with the global flag moved back in front of it.

    Rebuilt from the command names Click resolved, never from the raw argv. The
    first version of this hint echoed every remaining token, which turned
    ``potpie config set <key> <secret> --json`` into a next-action line that
    printed the secret to stdout and into the JSON envelope — reopening, one
    parse-error earlier, the very leak `config set` redacts on its own success
    path. Command names are safe to repeat back; the values typed after them
    are not, so they collapse to an ellipsis the caller can refill themselves.

    A value-taking flag is spelled with its metavar rather than bare: the only
    invocation that still reaches this hint is one that left the value out, and
    ``re-run as: potpie --host status`` would hand back the same broken line.
    """
    spelling = _ROOT_FLAG_SPELLING.get(flag, flag)
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
        return f"put '{flag}' before the command: potpie {spelling} <command> ..."
    rest = [a for a in args if a != flag]
    tail = rest[rest.index(path[-1]) + 1 :] if path[-1] in rest else []
    # Click stops resolving at the group that owns the offending flag, so
    # `potpie graph --json catalog` leaves its subcommand unconsumed in argv.
    # Naming it back is safe precisely because it matched a declared command.
    subcommands = getattr(getattr(ctx, "command", None), "commands", {}) or {}
    if tail and tail[0] in subcommands:
        path.append(tail.pop(0))
    return f"re-run as: potpie {spelling} {' '.join(path)}{' ...' if tail else ''}"


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
        # A global flag typed after the command is *honoured*, not refused.
        # `NoSuchOption` is raised while Click is still parsing, so no command
        # body has run and nothing has been emitted: hoisting the flag in front
        # of the command and invoking the app again is the same command line,
        # spelled the one way the parser accepts. Bounded by the number of
        # global flags there are, because each pass fixes exactly the one Click
        # named and Click names them one at a time.
        attempts = len(_ROOT_ONLY_FLAGS)
        while True:
            try:
                exit_code = app(args, standalone_mode=False)
                break
            except ClickException as exc:
                flag = _misplaced_root_flag(exc)
                hoisted = _hoist_root_flag(flag, args) if flag is not None else None
                if hoisted is None or attempts <= 0:
                    raise
                attempts -= 1
                args = hoisted
    except (Abort, click.Abort):
        raise typer.Exit(code=1) from None
    except ClickException as exc:
        flag = _misplaced_root_flag(exc)
        if flag is not None:
            # Only the shapes `_hoist_root_flag` refuses to move reach here — a
            # value-taking global with no value left to take. Routed through
            # `fail` in *both* modes, unlike the usage errors below: this is the
            # one where Click's own text ("No such option: --host") names the
            # symptom and hides the cause — the flag exists, one position to the
            # left.
            fail(
                code="usage_error",
                message=(
                    f"'{flag}' is a global flag and must come before the "
                    f"command: potpie {_ROOT_FLAG_SPELLING.get(flag, flag)} "
                    "<command> ..."
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
