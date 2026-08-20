"""``Daemon`` - local host lifecycle.

The daemon shell is the local background process for lifecycle, IPC, health,
and logs. It is not the business layer. When ``in_process`` is true, the host
runs in the CLI process and reports synthetic liveness. When detached, the
daemon process runs ``potpie.daemon.main`` and serves HostShell RPC over loopback
HTTP.

Liveness and readiness are separate: the daemon can be live while a backend or
semantic index is not ready.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.core.lifecycle import DONE, SKIPPED, SetupPlan, StepResult
from potpie.runtime.controller import (
    DaemonBootSpec,
    DaemonController,
    DaemonLaunchSpec,
    DaemonProcessHandle,
)
from potpie.runtime.protocol import ProtocolTransportError
from potpie_context_engine import Failure, Success


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


class _RecordedDaemonProcess:
    """Process handle for a child created by an earlier CLI invocation."""

    def __init__(self, pid: int, *, poll_interval_s: float = 0.05) -> None:
        self._pid = pid
        self._poll_interval_s = poll_interval_s
        self._returncode: int | None = None

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def returncode(self) -> int | None:
        if self._returncode is None and not _pid_alive(self._pid):
            self._returncode = 0
        return self._returncode

    async def wait(self) -> int:
        while _pid_alive(self._pid):
            await asyncio.sleep(self._poll_interval_s)
        self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        os.kill(self._pid, signal.SIGTERM)

    def kill(self) -> None:
        os.kill(self._pid, signal.SIGKILL)


class _ReflectiveDaemonObserver:
    """Temporary readiness adapter for the currently shipped daemon runtime."""

    def __init__(self, *, home: Path, poll_interval_s: float = 0.05) -> None:
        self._home = home
        self._poll_interval_s = poll_interval_s

    @property
    def instance_id(self) -> None:
        return None

    async def wait_ready(
        self, process: DaemonProcessHandle, *, timeout_s: float
    ) -> Success[None] | Failure[ProtocolTransportError]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_code = "legacy_discovery_unavailable"
        while loop.time() < deadline:
            if process.returncode is not None:
                last_code = "legacy_daemon_exited"
                break
            discovery = _read_reflective_discovery(self._home)
            if discovery is not None and _discovery_matches_pid(
                discovery, process.pid
            ):
                base_url = str(discovery.get("base_url") or "")
                if base_url and await _reflective_health_ready(
                    base_url=base_url,
                    expected_pid=process.pid,
                ):
                    return Success(None)
                last_code = "legacy_health_unavailable"
            await asyncio.sleep(
                min(self._poll_interval_s, max(0.0, deadline - loop.time()))
            )
        return Failure(
            ProtocolTransportError(
                code=last_code,
                message="the shipped daemon did not report ready",
                retry_posture="safe",
            )
        )

    async def request_stop(self) -> Failure[ProtocolTransportError]:
        # The shipped reflective runtime has no typed control operation. The
        # controller therefore takes its bounded signal fallback until commit 18.
        return Failure(
            ProtocolTransportError(
                code="legacy_typed_shutdown_unavailable",
                message="the shipped daemon has no typed shutdown operation",
                retry_posture="unknown",
            )
        )

    async def close(self) -> None:
        return None


async def _reflective_health_ready(*, base_url: str, expected_pid: int) -> bool:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=0.5) as client:
            response = await client.get(f"{base_url.rstrip('/')}/health")
        if not 200 <= response.status_code < 300:
            return False
        payload = response.json()
        return bool(payload.get("ok")) and int(payload.get("pid")) == expected_pid
    except (OSError, ValueError, TypeError):
        return False
    except Exception:  # noqa: BLE001 - readiness remains a typed best-effort probe.
        return False


def _read_reflective_discovery(home: Path) -> dict[str, Any] | None:
    for path in (home / "discovery.json", home / "daemon.json"):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, dict):
            return raw
    return None


def _discovery_matches_pid(discovery: dict[str, Any], pid: int) -> bool:
    try:
        return int(discovery.get("pid")) == pid
    except (TypeError, ValueError):
        return False


@dataclass
class Daemon:
    """Local daemon lifecycle: in-process stand-in or detached process."""

    home: Path = field(default_factory=default_home)
    in_process: bool = True
    startup_timeout_s: float = 60.0
    _controller: DaemonController | None = field(
        default=None, init=False, repr=False
    )
    _runner: asyncio.Runner | None = field(default=None, init=False, repr=False)

    def discovery(self) -> dict[str, str] | None:
        """Return daemon discovery metadata for either supported local daemon."""
        raw = _read_reflective_discovery(self.home)
        if raw is None:
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
        pid = self._recorded_pid()
        up = bool(pid and _pid_alive(pid))
        if pid is not None and not up:
            self._cleanup_runtime_records()
            pid = None
        discovery = self.discovery() or {}
        bind = discovery.get("bind", "")
        socket = bind.removeprefix("unix:") if bind.startswith("unix:") else bind
        base_url = discovery.get("base_url", "")
        controller_status = None
        if up and pid is not None:
            controller = self._controller_for_existing(pid)
            controller_status = self._run(controller.status())
        status = {
            "up": up,
            "ready": bool(controller_status and controller_status.ready),
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
        from potpie.daemon.process.ipc_client import client_for

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
        del follow  # Streaming remains a presentation concern; output is unchanged.
        controller = self._controller or self._new_controller(backend=None)
        return controller.logs()

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
        from potpie.daemon.process.launcher import DaemonStartError

        pid = self._recorded_pid()
        if pid is not None and _pid_alive(pid):
            raise DaemonStartError(f"daemon already running (pid={pid})")
        self._cleanup_runtime_records()
        controller = self._new_controller(backend=backend)
        outcome = self._run(controller.start())
        if isinstance(outcome, Failure):
            self._cleanup_stale_runtime_records()
            raise DaemonStartError(
                outcome.error.message,
                log_path=self.home / "logs" / "potpied.log",
            )
        self._controller = controller
        discovery = self.discovery() or {}
        return {
            "pid": outcome.value.pid,
            "socket": "",
            "bind": "",
            "url": discovery.get("base_url", ""),
        }

    def stop(self) -> dict[str, Any]:
        controller = self._controller
        pid = controller.pid if controller is not None else self._recorded_pid()
        if pid is None:
            self._cleanup_runtime_records()
            return {"detail": "daemon not running"}
        if not _pid_alive(pid):
            self._cleanup_runtime_records()
            return {"detail": "stale pid file removed"}
        discovery = _read_reflective_discovery(self.home)
        if discovery is not None and not _discovery_matches_pid(discovery, pid):
            self._cleanup_runtime_records()
            return {"detail": "stale daemon discovery removed"}
        if controller is None:
            controller = self._controller_for_existing(pid)
        outcome = self._run(controller.stop())
        self._cleanup_runtime_records()
        self._controller = None
        if isinstance(outcome, Failure):
            return {"detail": outcome.error.message}
        detail = {
            "already_stopped": "daemon not running",
            "typed_shutdown": "daemon stopped",
            "terminated": "daemon stopped",
            "killed": "daemon killed (forced after timeout)",
        }[outcome.value.mode]
        return {"detail": detail}

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

    def _new_controller(self, *, backend: str | None) -> DaemonController:
        return DaemonController(
            boot_factory=lambda: self._boot_spec(backend=backend),
            readiness_timeout_s=self.startup_timeout_s,
            stop_timeout_s=10.0,
            log_paths=(
                self.home / "logs" / "potpied.log",
                self.home / "daemon.log",
            ),
        )

    def _controller_for_existing(self, pid: int) -> DaemonController:
        if self._controller is not None and self._controller.pid == pid:
            return self._controller
        controller = self._new_controller(backend=None)
        process = _RecordedDaemonProcess(pid)
        launch = self._launch_spec(backend=None)
        self._run(
            controller.attach(
                process=process,
                observer=_ReflectiveDaemonObserver(home=self.home),
                launch=launch,
            )
        )
        self._controller = controller
        return controller

    def _run(self, awaitable: Any) -> Any:
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(awaitable)

    def _boot_spec(self, *, backend: str | None) -> DaemonBootSpec:
        return DaemonBootSpec(
            launch=self._launch_spec(backend=backend),
            observer=_ReflectiveDaemonObserver(home=self.home),
        )

    def _launch_spec(self, *, backend: str | None) -> DaemonLaunchSpec:
        from potpie.cli.telemetry.settings import load_cli_runtime_settings
        from potpie_context_engine.bootstrap.runtime_settings import (
            project_child_environment,
        )

        environment = project_child_environment(
            load_cli_runtime_settings(),
            os.environ,
            overrides={
                "CONTEXT_ENGINE_HOME": str(self.home),
                **({"CONTEXT_ENGINE_BACKEND": backend} if backend else {}),
            },
        )
        return DaemonLaunchSpec(
            command=(sys.executable, "-m", "potpie.daemon.main"),
            environment=environment,
            log_path=self.home / "logs" / "potpied.log",
        )

    def _recorded_pid(self) -> int | None:
        from potpie.daemon.process.pidfile import read_pid_file

        return read_pid_file(self.home / "daemon.pid")

    def _cleanup_stale_runtime_records(self) -> None:
        pid = self._recorded_pid()
        if pid is None or not _pid_alive(pid):
            self._cleanup_runtime_records()

    def _cleanup_runtime_records(self) -> None:
        from potpie.daemon.process.pidfile import remove_pid_file

        for name in ("daemon.pid", "discovery.json", "daemon.json"):
            remove_pid_file(self.home / name)


__all__ = ["Daemon"]
