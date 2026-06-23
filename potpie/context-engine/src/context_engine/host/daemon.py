"""``Daemon`` - local host lifecycle.

The daemon shell is the local background process for lifecycle, IPC, health,
and logs. It is not the business layer. When ``in_process`` is true, the host
runs in the CLI process and reports synthetic liveness. When detached, the
daemon process runs ``context_engine.host.daemon_main`` and serves
HostShell RPC over loopback HTTP.

Liveness and readiness are separate: the daemon can be live while a backend or
semantic index is not ready.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_engine.adapters.outbound.pots.local_pot_store import default_home
from context_engine.domain.lifecycle import DONE, SKIPPED, SetupPlan, StepResult


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
    startup_timeout_s: float = 60.0

    def discovery(self) -> dict[str, str] | None:
        """Return daemon discovery metadata for either supported local daemon."""
        from context_engine.adapters.outbound.daemon_process.ipc_client import load_discovery

        discovery = load_discovery(self.home)
        if discovery is not None:
            return discovery
        legacy_path = self.home / "daemon.json"
        if not legacy_path.exists():
            return None
        try:
            raw = json.loads(legacy_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(raw, dict):
            return None
        return {str(key): str(value) for key, value in raw.items()}

    def status(self) -> dict[str, Any]:
        if self.in_process:
            return {
                "up": True,
                "mode": "in_process",
                "home": str(self.home),
                "detail": "in-process host; no detached daemon",
            }
        from context_engine.adapters.outbound.daemon_process.pidfile import read_pid_file

        pid = read_pid_file(self.home / "daemon.pid")
        up = bool(pid and _pid_alive(pid))
        discovery = self.discovery() or {}
        bind = discovery.get("bind", "")
        socket = bind.removeprefix("unix:") if bind.startswith("unix:") else bind
        base_url = discovery.get("base_url", "")
        status = {
            "up": up,
            "mode": "detached",
            "home": str(self.home),
            "pid": pid,
            "detail": "detached daemon running"
            if up
            else "detached daemon not running",
        }
        if socket:
            status["socket"] = socket
        if base_url:
            status["url"] = base_url
        if up:
            health = self.health()
            if "backend" in health:
                status["backend"] = health["backend"]
        return status

    def health(self) -> dict[str, Any]:
        if self.in_process:
            return {"live": True, "mode": "in_process"}
        discovery = self.discovery() or {}
        base_url = discovery.get("base_url")
        if base_url:
            try:
                import httpx

                response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=3.0)
                data = response.json()
                return {
                    "live": 200 <= response.status_code < 300,
                    "mode": "detached",
                    **data,
                }
            except Exception:  # noqa: BLE001 - daemon health must be best-effort.
                return {"live": False, "mode": "detached"}
        from context_engine.adapters.outbound.daemon_process.ipc_client import client_for

        try:
            with client_for(self.home) as client:
                response = client.get("/admin/health")
                return {
                    "live": 200 <= response.status_code < 300,
                    "mode": "detached",
                    **response.json(),
                }
        except Exception:  # noqa: BLE001 - daemon health must be best-effort.
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
                SKIPPED,
                f"daemon already running (pid={status.get('pid')})",
                metadata={
                    "mode": "detached",
                    "pid": status.get("pid"),
                    "socket": status.get("socket"),
                    "url": status.get("url"),
                },
            )
        info = self.start(backend=plan.backend if plan is not None else None)
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

    def start(self, *, backend: str | None = None) -> dict[str, Any]:
        from context_engine.adapters.outbound.daemon_process.launcher import start_detached

        return start_detached(
            self.home,
            ready_timeout_s=self.startup_timeout_s,
            backend=backend,
        )

    def stop(self) -> dict[str, Any]:
        from context_engine.adapters.outbound.daemon_process.launcher import stop_daemon

        return {"detail": stop_daemon(self.home)}

    def restart(self) -> dict[str, Any]:
        current_status = self.status()
        current_backend = current_status.get("backend")
        if current_status.get("up") and not isinstance(current_backend, str):
            raise RuntimeError(
                "cannot determine running daemon backend; "
                "refusing restart to avoid backend drift"
            )
        self.stop()
        info = self.start(
            backend=current_backend if isinstance(current_backend, str) else None
        )
        status = self.status()
        if "backend" in status:
            info = {**info, "backend": status["backend"]}
        return {**info, "started": info}


__all__ = ["Daemon"]
