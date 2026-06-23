"""Service admin commands → the running daemon's ``/admin/services`` IPC (managed services).

Only meaningful when a detached daemon is running (``potpie daemon start`` /
``potpie setup --daemon``); the managed services live inside that process.
"""

from __future__ import annotations

import typer

from potpie.context_engine.adapters.inbound.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    emit,
    fail,
    get_host,
)
from potpie.context_engine.adapters.outbound.daemon_process.ipc_client import client_for

service_app = typer.Typer(help="Manage the daemon's supporting services.")


def _home():
    return get_host().daemon.home


def _client():
    """Open a client to the running daemon, or fail cleanly if it is not up."""
    try:
        return client_for(_home())
    except RuntimeError as exc:
        fail(
            code="daemon_not_running",
            message=str(exc),
            next_action="start it with 'potpie daemon start' (or 'potpie setup --daemon')",
            exit_code=EXIT_UNAVAILABLE,
        )


@service_app.command("up")
def service_up(name: str) -> None:
    with contract():
        with _client() as c:
            r = c.post(f"/admin/services/{name}/up")
            r.raise_for_status()
            ep = r.json().get("endpoint")
            emit({"name": name, "endpoint": ep}, human=f"up: {ep}")


@service_app.command("down")
def service_down(name: str) -> None:
    with contract():
        with _client() as c:
            r = c.post(f"/admin/services/{name}/down")
            r.raise_for_status()
            emit({"name": name, "ok": True}, human="down")


@service_app.command("status")
def service_status() -> None:
    with contract():
        with _client() as c:
            r = c.get("/admin/services")
            r.raise_for_status()
            services = r.json()["services"]
            emit(
                {"services": services},
                human="\n".join(
                    f"{s['name']:<24} {s['status']:<10} {s['endpoint'] or ''}"
                    for s in services
                )
                or "(no services)",
            )


@service_app.command("logs")
def service_logs(
    name: str, follow: bool = typer.Option(False, "-f", "--follow")
) -> None:
    with contract():
        log_path = _home() / "logs" / f"service-{name}.log"
        if not log_path.exists():
            emit({"lines": []}, human="no log file")
            return
        if not follow:
            text = log_path.read_text(errors="replace")
            emit({"lines": text.splitlines()}, human=text)
            return
        import time as _t

        with log_path.open() as f:
            f.seek(0, 2)
            try:
                while True:
                    line = f.readline()
                    if not line:
                        _t.sleep(0.2)
                        continue
                    typer.echo(line.rstrip("\n"))
            except KeyboardInterrupt:
                return


__all__ = ["service_app"]
