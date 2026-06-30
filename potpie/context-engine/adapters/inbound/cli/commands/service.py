"""Service admin commands → the running daemon's ``/admin/services`` IPC (managed services).

Only meaningful when a detached daemon is running (``potpie daemon start`` /
``potpie setup --daemon``); the managed services live inside that process.
"""

from __future__ import annotations

from typing import Any

import typer

from adapters.inbound.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    emit,
    fail,
    get_host,
)
from adapters.outbound.daemon_process.ipc_client import client_for
from adapters.outbound.settings_env import context_engine_falkordb_lite_path

service_app = typer.Typer(help="Manage the daemon's supporting services.")

_EMBEDDED_GRAPH_PROFILES = frozenset({"falkordb_lite", "embedded", "in_memory"})


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


def _normalize_service_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "falkordblite":
        return "falkordb_lite"
    return normalized


def _managed_service_names() -> set[str] | None:
    try:
        with client_for(_home()) as client:
            response = client.get("/admin/services")
            response.raise_for_status()
    except RuntimeError:
        return None
    return {
        str(service.get("name") or "")
        for service in response.json().get("services", ())
        if service.get("name")
    }


def _embedded_graph_profile(name: str) -> str | None:
    normalized = _normalize_service_name(name)
    if normalized in _EMBEDDED_GRAPH_PROFILES:
        return normalized
    return None


def _missing_service_log_response(name: str) -> tuple[dict[str, Any], str]:
    embedded_profile = _embedded_graph_profile(name)
    if embedded_profile is not None:
        detail = [
            (
                f"{embedded_profile} runs embedded inside the daemon "
                "(no separate service log)."
            ),
            "Use `potpie daemon logs` for daemon output.",
        ]
        payload: dict[str, Any] = {
            "lines": [],
            "status": "embedded_backend",
            "profile": embedded_profile,
            "recommended_log_command": "potpie daemon logs",
            "detail": detail,
        }
        if embedded_profile == "falkordb_lite":
            db_path = context_engine_falkordb_lite_path()
            detail.append(f"Database file: {db_path}")
            payload["database_path"] = db_path
        payload["message"] = detail[0]
        return payload, "\n".join(detail)

    managed = _managed_service_names()
    if managed is not None and name in managed:
        message = (
            f"no log file for managed service {name!r} yet "
            "(start it with `potpie service up` to create one)"
        )
    else:
        message = (
            f"no log file for {name!r}. "
            "Managed subprocess services write to "
            f"<home>/logs/service-<name>.log; run `potpie service status` to list them. "
            "For the default embedded graph backend, use `potpie daemon logs` instead."
        )
    return {"lines": [], "status": "no_log_file", "message": message}, message


@service_app.command("logs")
def service_logs(
    name: str, follow: bool = typer.Option(False, "-f", "--follow")
) -> None:
    with contract():
        log_path = _home() / "logs" / f"service-{name}.log"
        if not log_path.exists():
            payload, human = _missing_service_log_response(name)
            emit(payload, human=human)
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
