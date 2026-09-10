"""Daemon admin commands through the Potpie-owned lifecycle service."""

from __future__ import annotations

from typing import Any, NoReturn

import typer

from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    emit,
    fail,
)
from potpie.daemon.lifecycle import Daemon, DaemonStartError, DaemonStopError

daemon_app = typer.Typer(help="Local daemon lifecycle (recovery tools).")


def _detached_daemon() -> Daemon:
    return Daemon(in_process=False)


def _start(daemon: Daemon) -> dict[str, int | str]:
    try:
        return daemon.start()
    except DaemonStartError as exc:
        _fail_start(exc)


def _fail_start(exc: DaemonStartError) -> NoReturn:
    fail(
        code="daemon_start_failed",
        message=str(exc),
        detail=(str(exc.log_path) if exc.log_path else None),
        next_action=(
            getattr(exc, "recommended_next_action", None)
            or "inspect the daemon log with 'potpie daemon logs'"
        ),
        exit_code=EXIT_UNAVAILABLE,
    )


def _restart(daemon: Daemon) -> dict[str, int | str]:
    try:
        return daemon.restart()
    except AttributeError:
        _stop(daemon)
        return _start(daemon)
    except DaemonStartError as exc:
        _fail_start(exc)
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


def _stop(daemon: Daemon, *, force: bool = False) -> dict[str, Any]:
    try:
        # Only ask for the forced path when forcing, so any Daemon-shaped
        # collaborator keeps working with the plain signature.
        return daemon.stop(force=True) if force else daemon.stop()
    except DaemonStopError as exc:
        _fail_stop(exc)


@daemon_app.command("start")
def daemon_start() -> None:
    with contract():
        info = _start(_detached_daemon())
        emit(info, human=f"daemon started (pid={info.get('pid')})")


@daemon_app.command("status")
def daemon_status() -> None:
    with contract():
        st = _detached_daemon().status()
        emit(st, human=_status_human(st))


def _status_human(st: dict[str, Any]) -> str:
    lines = [f"daemon: {st['mode']} (up={st['up']})"]
    # A running daemon this build cannot authenticate is the single state that
    # wedges every other command, so name it here instead of reporting a bare
    # "up=True" that contradicts `potpie status`.
    if st.get("identity") == "unauthenticated":
        lines.append(f"  identity: unauthenticated — {st.get('detail')}")
        if st.get("recovery"):
            lines.append(f"  recovery: {st['recovery']}")
    return "\n".join(lines)


@daemon_app.command("logs")
def daemon_logs(follow: bool = typer.Option(False, "--follow")) -> None:
    with contract():
        lines = _detached_daemon().logs(follow=follow)
        emit({"lines": lines}, human="\n".join(lines) or "(no logs)")


@daemon_app.command("restart")
def daemon_restart() -> None:
    with contract():
        daemon = _detached_daemon()
        info = _restart(daemon)
        emit(info, human=f"restarted (pid={info.get('pid')})")


@daemon_app.command("stop")
def daemon_stop(
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Replace a daemon whose runtime record cannot be authenticated "
            "(verifies the recorded process is a potpie daemon first)."
        ),
    ),
) -> None:
    with contract():
        result = _stop(_detached_daemon(), force=force)
        emit(result, human=result.get("detail", "stopped"))


__all__ = ["daemon_app"]
