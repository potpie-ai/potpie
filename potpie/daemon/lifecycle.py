"""``Daemon`` - active local HTTP/RPC daemon lifecycle.

The daemon shell is the local background process for lifecycle, IPC, health,
and logs. It is not the business layer. When ``in_process`` is true, the host
runs in the CLI process and reports synthetic liveness. When detached, the
daemon process runs ``potpie.daemon.main`` and serves HostShell RPC over loopback
HTTP. Legacy UDS operation-runtime helpers are intentionally not part of this
lifecycle contract.

Liveness and readiness are separate: the daemon can be live while a backend or
semantic index is not ready.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.domain.lifecycle import DONE, SKIPPED, SetupPlan, StepResult
from potpie_context_engine.domain.ports.daemon.lifecycle import (
    DaemonDiscovery,
    DaemonHealth,
    DaemonInstallResult,
    DaemonRestartResult,
    DaemonStartResult,
    DaemonStatus,
    DaemonStopResult,
)


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

    def discovery(self) -> DaemonDiscovery | None:
        """Return active HTTP/RPC daemon discovery metadata."""
        from potpie.daemon.process.ipc_client import load_discovery

        return load_discovery(self.home)

    def status(self) -> DaemonStatus:
        if self.in_process:
            return {
                "up": True,
                "mode": "in_process",
                "home": str(self.home),
                "detail": "in-process host; no detached daemon",
            }
        from potpie.daemon.process.pidfile import read_pid_file

        pid = read_pid_file(self.home / "daemon.pid")
        up = bool(pid and _pid_alive(pid))
        discovery = self.discovery() or {}
        base_url = discovery.get("base_url", "")
        status: DaemonStatus = {
            "up": up,
            "mode": "detached",
            "home": str(self.home),
            "pid": pid,
            "detail": "detached daemon running"
            if up
            else "detached daemon not running",
        }
        if base_url:
            status["url"] = base_url
        if up:
            health = self.health()
            if "backend" in health:
                status["backend"] = health["backend"]
        return status

    def health(self) -> DaemonHealth:
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
        return {"live": False, "mode": "detached"}

    def _log_file(self) -> Path | None:
        for name in ("logs/potpied.log", "daemon.log"):
            log_file = self.home / name
            if log_file.exists():
                return log_file
        return None

    def logs(self) -> list[str]:
        log_file = self._log_file()
        if log_file is not None:
            return log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return []

    def iter_logs(self, *, poll_interval_s: float = 0.2) -> Iterator[str]:
        log_file = self._log_file()
        if log_file is None:
            return
        with log_file.open(encoding="utf-8", errors="replace") as stream:
            while True:
                line = stream.readline()
                if not line:
                    time.sleep(poll_interval_s)
                    continue
                yield line.rstrip("\n")

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

    def install(self) -> DaemonInstallResult:
        return {
            "installed": False,
            "detail": "no service unit for the local OSS daemon",
        }

    def start(self, *, backend: str | None = None) -> DaemonStartResult:
        from potpie.daemon.process.launcher import start_detached

        return start_detached(
            self.home,
            ready_timeout_s=self.startup_timeout_s,
            backend=backend,
        )

    def stop(self) -> DaemonStopResult:
        from potpie.daemon.process.launcher import stop_daemon

        return {"detail": stop_daemon(self.home)}

    def restart(self) -> DaemonRestartResult:
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
