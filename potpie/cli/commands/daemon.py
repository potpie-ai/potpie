"""Daemon admin commands through the Potpie-owned lifecycle service."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, NoReturn

import typer

from potpie import build_info
from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    emit,
    fail,
    is_json,
    json_error_formatter,
)
from potpie.daemon.lifecycle import Daemon, DaemonStartError, DaemonStopError
from potpie.daemon.log_reader import DEFAULT_LOG_TAIL_LINES, parse_since

daemon_app = typer.Typer(help="Local daemon lifecycle (recovery tools).")


def _detached_daemon() -> Daemon:
    return Daemon(in_process=False)


def _start(daemon: Daemon) -> dict[str, int | str]:
    try:
        return daemon.start()
    except DaemonStartError as exc:
        fail(
            code="daemon_start_failed",
            message=str(exc),
            detail=(str(exc.log_path) if exc.log_path else None),
            next_action="inspect the daemon log with 'potpie daemon logs'",
            exit_code=EXIT_UNAVAILABLE,
        )


def _restart(daemon: Daemon) -> dict[str, int | str]:
    try:
        return daemon.restart()
    except AttributeError:
        _stop(daemon)
        return _start(daemon)
    except DaemonStartError as exc:
        fail(
            code="daemon_start_failed",
            message=str(exc),
            detail=(str(exc.log_path) if exc.log_path else None),
            next_action="inspect the daemon log with 'potpie daemon logs'",
            exit_code=EXIT_UNAVAILABLE,
        )
    except DaemonStopError as exc:
        _fail_stop(exc)


def _fail_stop(exc: DaemonStopError) -> NoReturn:
    error = exc.error
    fail(
        code=error.code,
        message=error.message,
        detail=error.details or None,
        next_action=error.recommended_next_action,
        exit_code=EXIT_UNAVAILABLE,
    )


def _stop(daemon: Daemon) -> dict[str, Any]:
    try:
        return daemon.stop()
    except DaemonStopError as exc:
        _fail_stop(exc)


@daemon_app.command("start")
def daemon_start() -> None:
    with contract():
        info = _start(_detached_daemon())
        emit(info, human=f"daemon started (pid={info.get('pid')})")


@daemon_app.command("status")
def daemon_status() -> None:
    """Report whether the daemon serves, and exit non-zero when it does not.

    Exit 0 means the daemon answered an authenticated handshake, so
    ``potpie daemon status || potpie daemon start`` reads as it should. A
    daemon that is down, or whose process exists but does not answer, is
    ``daemon_unavailable`` (exit 2). The JSON payload still carries the full
    status under the error keys, so a caller gating on the exit code can
    report what it found.
    """

    with contract():
        st = _detached_daemon().status()
        if st.get("up") and st.get("ready"):
            emit(st, human=_status_human(st))
            return
        if st.get("up"):
            message = (
                f"detached daemon process {st.get('pid')} exists but is not "
                "answering (stopped, wedged, or still starting)"
            )
            next_action = "restart it with 'potpie daemon restart'"
        else:
            message = str(st.get("detail") or "detached daemon not running")
            next_action = "start it with 'potpie daemon start'"
        with json_error_formatter(lambda payload: {**st, **payload}):
            fail(
                code="daemon_unavailable",
                message=message,
                detail=st.get("detail"),
                next_action=next_action,
                exit_code=EXIT_UNAVAILABLE,
            )


def _status_human(st: dict[str, Any]) -> str:
    lines = [f"daemon: {st['mode']} (up={st['up']})"]
    build = st.get("build")
    if isinstance(build, dict):
        lines.append(f"  build: {_build_note(build, st.get('stale'))}")
    return "\n".join(lines)


def _build_note(build: dict[str, Any], stale: object) -> str:
    rev = build_info.short_rev(build.get("rev")) or "rev unknown"
    if build.get("dirty"):
        rev = f"{rev}, dirty"
    if stale is True:
        ours = build_info.short_rev(build_info.build_stamp().get("rev"))
        return (
            f"{rev} (stale: this CLI is {ours}; "
            "run 'potpie daemon restart' to serve this build)"
        )
    if stale is False:
        return f"{rev} (current)"
    return rev


@daemon_app.command("logs")
def daemon_logs(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Stream new lines until interrupted."
    ),
    tail: int = typer.Option(
        DEFAULT_LOG_TAIL_LINES,
        "--tail",
        "-n",
        help="How many trailing lines to show; 0 for the whole file.",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Only lines at or after this time: ISO-8601, or an age like 15m.",
    ),
) -> None:
    """Show the end of the daemon log.

    Bounded by default: nothing rotates the log, so an unbounded read grows
    with every line the daemon has ever written.
    """

    with contract():
        if tail < 0:
            raise ValueError("--tail must be zero or a positive number of lines")
        cutoff = parse_since(since) if since else None
        daemon = _detached_daemon()
        limit = tail if tail > 0 else None
        if not follow:
            lines = daemon.logs(tail=limit, since=cutoff)
            log_file = daemon.log_path()
            emit(
                {
                    "lines": lines,
                    "log_file": str(log_file) if log_file is not None else None,
                    "follow": False,
                },
                human="\n".join(lines) or "(no logs)",
            )
            return
        _stream(daemon.follow_logs(tail=limit, since=cutoff))


def _stream(lines: Iterable[str]) -> None:
    """Print a follow stream one line at a time until interrupted.

    ``--json`` prints one JSON object per line: a stream that never ends cannot
    be one JSON document. Ctrl-C is how a follow ends, so it is a success.
    """

    json_mode = is_json()
    try:
        for line in lines:
            typer.echo(json.dumps({"line": line}) if json_mode else line)
    except KeyboardInterrupt:  # pragma: no cover - interactive exit
        pass


@daemon_app.command("restart")
def daemon_restart() -> None:
    with contract():
        daemon = _detached_daemon()
        info = _restart(daemon)
        emit(info, human=f"restarted (pid={info.get('pid')})")


@daemon_app.command("stop")
def daemon_stop() -> None:
    with contract():
        result = _stop(_detached_daemon())
        emit(result, human=result.get("detail", "stopped"))


__all__ = ["daemon_app"]
