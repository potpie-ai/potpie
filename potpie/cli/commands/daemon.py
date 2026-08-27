"""Daemon admin commands through the Potpie-owned lifecycle service."""

from __future__ import annotations

from typing import Any, NoReturn

import typer

from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    emit,
    fail,
    get_daemon_service,
)
from potpie.daemon.lifecycle import Daemon, DaemonStartError, DaemonStopError

daemon_app = typer.Typer(help="Local daemon lifecycle (recovery tools).")


def _detached_daemon() -> Daemon:
    daemon = get_daemon_service()
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
    with contract():
        st = _detached_daemon().status()
        emit(st, human=f"daemon: {st['mode']} (up={st['up']})")


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
def daemon_stop() -> None:
    with contract():
        result = _stop(_detached_daemon())
        emit(result, human=result.get("detail", "stopped"))


__all__ = ["daemon_app"]
