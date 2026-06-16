"""``Daemon`` - local host lifecycle.

The daemon shell is the local background process for lifecycle, IPC, health,
and logs. It is not the business layer. When ``in_process`` is true, the host
runs in the CLI process and reports synthetic liveness. When detached, the
daemon process runs ``host.daemon_runtime`` and serves HostShell operations over
a Unix socket.

Liveness and readiness are separate: the daemon can be live while a backend or
semantic index is not ready.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.outbound.pots.local_pot_store import default_home
from domain.lifecycle import DONE, SKIPPED, SetupPlan, StepResult


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


@dataclass
class Daemon:
    """Local daemon lifecycle: in-process stand-in or detached process."""

    home: Path = field(default_factory=default_home)
    in_process: bool = True

    def status(self) -> dict[str, Any]:
        if self.in_process:
            return {
                "up": True,
                "mode": "in_process",
                "home": str(self.home),
                "detail": "in-process host; no detached daemon",
            }
        from adapters.outbound.daemon_process.ipc_client import load_discovery
        from adapters.outbound.daemon_process.pidfile import read_pid_file

        pid = read_pid_file(self.home / "daemon.pid")
        up = bool(pid and _pid_alive(pid))
        discovery = load_discovery(self.home) or {}
        bind = discovery.get("bind", "")
        socket = bind.removeprefix("unix:") if bind.startswith("unix:") else bind
        return {
            "up": up,
            "mode": "detached",
            "home": str(self.home),
            "pid": pid,
            "socket": socket,
            "detail": "detached daemon running"
            if up
            else "detached daemon not running",
        }

    def health(self) -> dict[str, Any]:
        if self.in_process:
            return {"live": True, "mode": "in_process"}
        from adapters.outbound.daemon_process.ipc_client import client_for

        try:
            with client_for(self.home) as client:
                response = client.get("/admin/health")
                return {
                    "live": 200 <= response.status_code < 300,
                    "mode": "detached",
                    **response.json(),
                }
        except RuntimeError:
            return {"live": False, "mode": "detached"}

    def logs(self, *, follow: bool = False) -> list[str]:
        for name in ("logs/potpied.log", "daemon.log"):
            log_file = self.home / name
            if log_file.exists():
                return log_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
        return []

    def ensure(self, plan: SetupPlan | None = None) -> StepResult:
        if self.in_process:
            return StepResult(
                "daemon.ensure",
                SKIPPED,
                "in-process host; no detached daemon to start",
                metadata={"mode": "in_process"},
            )
        status = self.status()
        if status["up"]:
            return StepResult(
                "daemon.ensure",
                DONE,
                f"daemon already running (pid={status.get('pid')})",
                metadata={
                    "mode": "detached",
                    "pid": status.get("pid"),
                    "socket": status.get("socket"),
                },
            )
        info = self.start()
        return StepResult(
            "daemon.ensure",
            DONE,
            f"daemon started (pid={info.get('pid')})",
            metadata={"mode": "detached", **info},
        )

    def install(self) -> dict[str, Any]:
        return {
            "installed": False,
            "detail": "no service unit for the local OSS daemon",
        }

    def start(self) -> dict[str, Any]:
        from adapters.outbound.daemon_process.launcher import start_detached

        return start_detached(self.home)

    def stop(self) -> dict[str, Any]:
        from adapters.outbound.daemon_process.launcher import stop_daemon

        return {"detail": stop_daemon(self.home)}

    def restart(self) -> dict[str, Any]:
        self.stop()
        return self.start()


__all__ = ["Daemon"]
