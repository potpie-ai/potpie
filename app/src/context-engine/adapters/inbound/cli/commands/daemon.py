"""Daemon admin commands → ``HostShell.daemon`` (local recovery tools)."""

from __future__ import annotations

import typer

from adapters.inbound.cli.commands._common import contract, emit, get_host

daemon_app = typer.Typer(help="Local daemon lifecycle (recovery tools).")


@daemon_app.command("status")
def daemon_status() -> None:
    with contract():
        st = get_host().daemon.status()
        emit(st, human=f"daemon: {st['mode']} (up={st['up']})")


@daemon_app.command("logs")
def daemon_logs(follow: bool = typer.Option(False, "--follow")) -> None:
    with contract():
        lines = get_host().daemon.logs(follow=follow)
        emit({"lines": lines}, human="\n".join(lines) or "(no logs)")


@daemon_app.command("restart")
def daemon_restart() -> None:
    with contract():
        emit(get_host().daemon.restart(), human="restarted")


@daemon_app.command("stop")
def daemon_stop() -> None:
    with contract():
        emit(get_host().daemon.stop(), human="stopped")


__all__ = ["daemon_app"]
