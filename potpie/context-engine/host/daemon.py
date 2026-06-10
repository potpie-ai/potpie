"""``Daemon`` — local host lifecycle.

The daemon shell is the local background process for lifecycle, auth, IPC,
health, and logs — it is *not* the business layer. Production CLI paths use this
object as a lifecycle controller for the detached process; tests and explicit dev
mode can still use the in-process host.

Liveness and readiness are separate (see observability.md): the daemon can be
live while a backend or semantic index is not ready.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from adapters.outbound.daemon_process.pidfile import read_discovery, read_pid_file
from adapters.outbound.pots.local_pot_store import default_home
from domain.errors import CapabilityNotImplemented
from domain.lifecycle import DONE, SKIPPED, SetupPlan, StepResult


@dataclass(slots=True)
class Daemon:
    """Local daemon lifecycle controller.

    ``in_process=True`` keeps the old direct-host behavior for focused tests and
    development. Normal CLI wiring uses ``in_process=False`` and controls a
    background daemon process through pid/discovery files.
    """

    home: Path = field(default_factory=default_home)
    in_process: bool = True
    startup_timeout_s: float = 10.0

    @property
    def pid_file(self) -> Path:
        return self.home / "daemon.pid"

    @property
    def discovery_file(self) -> Path:
        return self.home / "daemon.json"

    @property
    def log_file(self) -> Path:
        return self.home / "daemon.log"

    def status(self) -> dict[str, Any]:
        if self.in_process:
            return {
                "up": True,
                "mode": "in_process",
                "home": str(self.home),
                "detail": "in-process host; detached daemon lifecycle disabled",
            }
        discovery = self.discovery()
        pid = read_pid_file(self.pid_file)
        if discovery is None or pid is None:
            return {
                "up": False,
                "mode": "detached",
                "home": str(self.home),
                "detail": "daemon is not running",
            }
        health = self._health(discovery)
        if health is None:
            return {
                "up": False,
                "mode": "detached",
                "home": str(self.home),
                "pid": pid,
                "detail": "discovery exists but health probe failed",
            }
        return {
            "up": True,
            "mode": "detached",
            "home": str(self.home),
            "pid": pid,
            "base_url": discovery.get("base_url"),
            "backend": health.get("backend"),
            "detail": "daemon is running",
        }

    def health(self) -> dict[str, Any]:
        # Liveness only — readiness is the services'/backend's concern.
        if self.in_process:
            return {"live": True, "mode": "in_process"}
        discovery = self.discovery()
        health = self._health(discovery) if discovery else None
        return {"live": bool(health), "mode": "detached", "detail": health}

    def logs(self, *, follow: bool = False) -> list[str]:
        log_file = self.log_file
        if not log_file.exists():
            return []
        return log_file.read_text(encoding="utf-8").splitlines()

    def ensure(self, plan: SetupPlan | None = None) -> StepResult:
        """Idempotent setup seam: make sure the host is running.

        In-process hosts have nothing to start, so this is a no-op that reports
        ``skipped``. A detached daemon owner fills this with install+start (idempotent).
        """
        if self.in_process:
            daemon_hosted = plan is not None and plan.host_mode == "daemon"
            return StepResult(
                step="daemon.ensure",
                state=SKIPPED,
                detail=(
                    "daemon already bootstrapped; setup is executing inside daemon host"
                    if daemon_hosted
                    else "in-process host; no detached daemon to start"
                ),
                metadata={"mode": "daemon_hosted" if daemon_hosted else "in_process"},
            )
        before = self.status()
        if before.get("up"):
            return StepResult(
                step="daemon.ensure",
                state=SKIPPED,
                detail=f"daemon already running (pid={before.get('pid')})",
                metadata={"mode": "detached", "pid": before.get("pid")},
            )
        started = self.start(plan=plan)
        return StepResult(
            step="daemon.ensure",
            state=DONE,
            detail=started.get("detail") or "daemon started",
            metadata={key: value for key, value in started.items() if key != "detail"},
        )

    def install(self) -> dict[str, Any]:
        if self.in_process:
            raise CapabilityNotImplemented(
                "host.daemon.install",
                detail="in-process host does not install a detached daemon",
            )
        return {"installed": True, "detail": "daemon runs as a user background process"}

    def start(self, *, plan: SetupPlan | None = None) -> dict[str, Any]:
        if self.in_process:
            raise CapabilityNotImplemented(
                "host.daemon.start",
                recommended_next_action="host runs in-process; no start needed",
            )
        current = self.status()
        if current.get("up"):
            return {**current, "started": False, "detail": "daemon already running"}

        self.home.mkdir(parents=True, exist_ok=True)
        source_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["CONTEXT_ENGINE_HOME"] = str(self.home)
        if plan is not None and plan.backend:
            env["CONTEXT_ENGINE_BACKEND"] = plan.backend
        env["PYTHONPATH"] = (
            f"{source_root}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(source_root)
        )
        log_fh = open(self.log_file, "ab")
        subprocess.Popen(
            [sys.executable, "-m", "host.daemon_main"],
            cwd=str(source_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
        status = self._wait_until_ready()
        if not status.get("up"):
            raise RuntimeError(status.get("detail") or "daemon did not become ready")
        return {**status, "started": True, "detail": "daemon started"}

    def stop(self) -> dict[str, Any]:
        if self.in_process:
            raise CapabilityNotImplemented(
                "host.daemon.stop",
                recommended_next_action="host runs in-process; nothing to stop",
            )
        pid = read_pid_file(self.pid_file)
        if pid is None:
            self._remove_runtime_files()
            return {"stopped": False, "detail": "daemon was not running"}
        if pid == os.getpid():
            raise RuntimeError("refusing to stop current process as daemon")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._remove_runtime_files()
            return {"stopped": False, "detail": "stale daemon pid removed"}
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                self._remove_runtime_files()
                return {"stopped": True, "pid": pid, "detail": "daemon stopped"}
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._remove_runtime_files()
        return {"stopped": True, "pid": pid, "detail": "daemon killed after timeout"}

    def restart(self) -> dict[str, Any]:
        if self.in_process:
            raise CapabilityNotImplemented(
                "host.daemon.restart",
                recommended_next_action="host runs in-process; nothing to restart",
            )
        before = self.status()
        backend = before.get("backend") if before.get("up") else None
        stopped = self.stop()
        started = self.start(plan=SetupPlan(backend=backend) if backend else None)
        return {"stopped": stopped, "started": started, "detail": "daemon restarted"}

    def discovery(self) -> dict[str, Any] | None:
        if not self.discovery_file.exists():
            return None
        try:
            return read_discovery(self.discovery_file)
        except Exception:
            return None

    def _wait_until_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.startup_timeout_s
        last = "waiting for daemon discovery"
        while time.monotonic() < deadline:
            status = self.status()
            if status.get("up"):
                return status
            last = str(status.get("detail") or last)
            time.sleep(0.1)
        return {"up": False, "mode": "detached", "detail": last}

    def _health(self, discovery: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = httpx.get(
                f"{str(discovery['base_url']).rstrip('/')}/health", timeout=1.0
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _remove_runtime_files(self) -> None:
        for path in (self.pid_file, self.discovery_file):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return not _pid_is_zombie(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _pid_is_zombie(pid: int) -> bool:
    if os.name != "posix":
        return False
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return False
    return result.stdout.strip().startswith("Z")


__all__ = ["Daemon"]
