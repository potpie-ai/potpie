"""Daemon admin commands -> ``HostShell.daemon`` local recovery tools."""

from __future__ import annotations

import json
from typing import Iterable

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_host_for,
    is_json,
    json_error_formatter,
)
from potpie.daemon.process.launcher import DaemonStartError
from potpie.daemon.lifecycle import (
    DEFAULT_LOG_TAIL_LINES,
    Daemon,
    parse_since,
)

daemon_app = typer.Typer(help="Local daemon lifecycle (recovery tools).")


def _detached_daemon() -> Daemon:
    """The local daemon process — never the active host.

    A managed host is a remote service with no process here to start, stop or
    tail, so "the daemon" in these commands can only ever mean the local one.
    Pinning it means `potpie daemon restart` still repairs your machine while
    the CLI is pointed at a managed pot, which is exactly when you are most
    likely to need it.
    """
    from potpie.cli import hosts

    daemon = get_host_for(hosts.LOCAL).daemon
    if not daemon.in_process:
        return daemon
    return Daemon(home=daemon.home, in_process=False)


def _start(daemon: Daemon) -> dict[str, int | str]:
    try:
        return daemon.start()
    except DaemonStartError as exc:
        fail(
            code="daemon_start_failed",
            message=str(exc),
            detail=(str(exc.log_path) if exc.log_path else None),
            next_action="inspect the daemon log with 'potpie daemon logs'",
        )


def _restart(daemon: Daemon) -> dict[str, int | str]:
    try:
        return daemon.restart()
    except AttributeError:
        daemon.stop()
        return _start(daemon)
    except DaemonStartError as exc:
        fail(
            code="daemon_start_failed",
            message=str(exc),
            detail=(str(exc.log_path) if exc.log_path else None),
            next_action="inspect the daemon log with 'potpie daemon logs'",
        )


@daemon_app.command("start")
def daemon_start() -> None:
    with contract():
        info = _start(_detached_daemon())
        emit(info, human=f"daemon started (pid={info.get('pid')})")


@daemon_app.command("status")
def daemon_status() -> None:
    """Report daemon liveness — and *exit* on it, so scripts can gate on it.

    This exited ``0`` whether the daemon was up, down, or wedged, which made the
    number worthless: the one command whose entire job is to answer "is it
    alive?" could only be consulted by parsing its output. A daemon that is not
    serving is ``daemon_unavailable`` — the code the exit table already maps to
    ``2`` for every other "a dependency did not answer", so ``potpie daemon
    status || potpie daemon start`` now means what it reads like.

    The payload is *not* reduced to an error envelope. ``--json`` still carries
    the full status (``state``, ``pid``, ``url``, ``home``) with the five error
    keys laid over it, because a caller that gates on the exit code usually
    wants to report what it found.

    ``doctor``'s always-zero rule does not apply here and is not a
    contradiction: ``doctor`` is the report you run *because* something is down,
    while this is the probe you run *to find out*.
    """
    with contract():
        st = _detached_daemon().status()
        human = f"daemon: {st['state']} (mode={st['mode']})"
        if st.get("serving"):
            emit(st, human=human)
            return
        with json_error_formatter(lambda payload: {**st, **payload}):
            fail(
                code="daemon_unavailable",
                message=str(st.get("detail") or human),
                next_action=(
                    "restart it with 'potpie daemon restart'"
                    if st.get("up")
                    else "start it with 'potpie daemon start'"
                ),
            )


@daemon_app.command("logs")
def daemon_logs(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="stream new lines until interrupted"
    ),
    tail: int = typer.Option(
        DEFAULT_LOG_TAIL_LINES,
        "--tail",
        "-n",
        help="how many trailing lines to show; 0 for the whole file",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="only lines at or after this time: ISO-8601, or an age like 15m",
    ),
) -> None:
    """Tail the daemon log.

    Bounded by default. Nothing rotates this file, so the old whole-file read
    returned every line the daemon had ever written — and ``--follow`` was
    accepted and then ignored, which is worse than rejecting it: the caller
    believes it is watching a live stream that ended before it was printed.
    """
    with contract():
        cutoff = parse_since(since) if since else None
        daemon = _detached_daemon()
        limit = tail if tail > 0 else None
        if not follow:
            lines = daemon.logs(tail=limit, since=cutoff)
            emit(
                {"lines": lines, "log_file": _log_file(daemon), "follow": False},
                human="\n".join(lines) or "(no logs)",
            )
            return
        _stream(daemon.follow_logs(tail=limit, since=cutoff))


def _log_file(daemon: Daemon) -> str | None:
    path = daemon.log_path()
    return str(path) if path is not None else None


def _stream(lines: Iterable[str]) -> None:
    """Print a follow stream one line at a time until the reader gives up.

    ``--json`` gets NDJSON — one object per line — because a stream that never
    ends cannot be one JSON document, and a half-written array is not something
    a consumer can parse incrementally. ``Ctrl-C`` is how this command is meant
    to end, so it is a success, not an aborted run.
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
        result = _detached_daemon().stop()
        emit(result, human=result.get("detail", "stopped"))


__all__ = ["daemon_app"]
