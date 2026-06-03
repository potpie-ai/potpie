"""``Daemon`` — local host lifecycle.

The daemon shell is the local background process for lifecycle, IPC, health, and logs —
it is *not* the business layer. When ``in_process`` (the default), the host runs in the
CLI process and the daemon reports a synthetic "in-process" liveness. When detached
(``host_mode = "daemon"``), the real bodies here drive
``adapters.outbound.daemon_process`` to start/stop a background process that runs
``host.daemon_runtime`` and serves the host's surfaces over a Unix socket.

Liveness and readiness are separate (see observability.md): the daemon can be live while
a backend or semantic index is not ready.
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
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


@dataclass(slots=True)
class Daemon:
    """Local daemon lifecycle: in-process stand-in by default, detached process when asked."""

    home: Path = field(default_factory=default_home)
    in_process: bool = True

    # --- introspection ------------------------------------------------------
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
        disc = load_discovery(self.home) or {}
        bind = disc.get("bind", "")
        socket = bind[len("unix:"):] if bind.startswith("unix:") else bind
        return {
            "up": up,
            "mode": "detached",
            "home": str(self.home),
            "pid": pid,
            "socket": socket,
            "detail": "detached daemon running" if up else "detached daemon not running",
        }

    def health(self) -> dict[str, Any]:
        # Liveness only — readiness is the services'/backend's concern.
        if self.in_process:
            return {"live": True, "mode": "in_process"}
        from adapters.outbound.daemon_process.ipc_client import client_for

        try:
            with client_for(self.home) as c:
                r = c.get("/admin/health")
                return {"live": r.status_code < 500, "mode": "detached", **r.json()}
        except Exception:
            return {"live": False, "mode": "detached"}

    def logs(self, *, follow: bool = False) -> list[str]:
        for name in ("logs/potpied.log", "daemon.log"):
            log_file = self.home / name
            if log_file.exists():
                return log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return []

    # --- setup seam ---------------------------------------------------------
    def ensure(self, plan: SetupPlan | None = None) -> StepResult:
        """Idempotent setup seam: make sure the host is running.

        In-process hosts have nothing to start (reports ``skipped``). A detached daemon
        starts the background process if it is not already running and reports ``done``.
        """
        if self.in_process:
            return StepResult(
                step="daemon.ensure",
                state=SKIPPED,
                detail="in-process host; no detached daemon to start",
                metadata={"mode": "in_process"},
            )
        st = self.status()
        if st["up"]:
            return StepResult(
                step="daemon.ensure",
                state=DONE,
                detail=f"daemon already running (pid={st.get('pid')})",
                metadata={"mode": "detached", "pid": st.get("pid"), "socket": st.get("socket")},
            )
        info = self.start()
        return StepResult(
            step="daemon.ensure",
            state=DONE,
            detail=f"daemon started (pid={info.get('pid')})",
            metadata={"mode": "detached", **info},
        )

    # --- lifecycle ----------------------------------------------------------
    def install(self) -> dict[str, Any]:
        # No OS service unit for the local OSS daemon; idempotent no-op so the
        # host-gated 'installer' setup step succeeds rather than gating the run.
        return {"installed": False, "detail": "no service unit for the local OSS daemon"}

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
