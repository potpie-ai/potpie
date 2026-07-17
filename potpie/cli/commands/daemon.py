"""Daemon admin commands -> ``HostShell.daemon`` local recovery tools."""

from __future__ import annotations

import typer

from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    emit,
    fail,
    get_host,
)
from potpie.daemon.process.launcher import DaemonStartError
from potpie.daemon.daemon import Daemon

daemon_app = typer.Typer(help="Local daemon lifecycle (recovery tools).")


def _detached_daemon() -> Daemon:
    daemon = get_host().daemon
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
        daemon.stop()
        return _start(daemon)
    except DaemonStartError as exc:
        fail(
            code="daemon_start_failed",
            message=str(exc),
            detail=(str(exc.log_path) if exc.log_path else None),
            next_action="inspect the daemon log with 'potpie daemon logs'",
            exit_code=EXIT_UNAVAILABLE,
        )


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
        result = _detached_daemon().stop()
        emit(result, human=result.get("detail", "stopped"))


__all__ = ["daemon_app"]
