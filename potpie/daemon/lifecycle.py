"""Canonical local daemon lifecycle controlled by Potpie."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from potpie.daemon.discovery import (
    DaemonDiscoveryError,
    canonical_discovery,
    load_daemon_connection,
    read_daemon_pid,
    read_daemon_discovery,
    remove_daemon_runtime_records,
    select_runtime_endpoint,
    write_daemon_discovery,
    write_daemon_pid,
)
from potpie.runtime.controller import (
    DaemonBootSpec,
    DaemonController,
    DaemonLaunchSpec,
    DaemonObserver,
)
from potpie.runtime.resource_manager import ResourceLifecycleError
from potpie.runtime.ownership import RuntimeOwnershipLock
from potpie.runtime.server import generate_bearer_token
from potpie.runtime.transport import RuntimeEndpoint
from potpie_context_engine import Failure, Success
from potpie.config.local_paths import default_home
from potpie_context_engine.core.lifecycle import DONE, SKIPPED, SetupPlan, StepResult


class DaemonStartError(Exception):
    """Raised when the canonical local daemon cannot become ready."""

    def __init__(self, message: str, *, log_path: Path | None = None) -> None:
        super().__init__(message)
        self.log_path = log_path


class DaemonStopError(Exception):
    """Raised when a detached daemon cannot be stopped safely."""

    def __init__(self, error: ResourceLifecycleError) -> None:
        super().__init__(error.message)
        self.error = error


def _pid_alive(pid: int) -> bool:
    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
    except OSError:
        # A daemon launched by another CLI process is not our child. Retain the
        # signal probe for that normal cross-process observation path.
        pass
    else:
        # ``kill(pid, 0)`` still succeeds for an exited child that is a zombie
        # on POSIX. Reap that child before deciding whether its PID is live.
        if waited_pid == pid:
            return False
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


@dataclass
class Daemon:
    """Local daemon lifecycle: in-process stand-in or detached process."""

    home: Path = field(default_factory=default_home)
    in_process: bool = True
    startup_timeout_s: float = 60.0
    _controller: DaemonController | None = field(default=None, init=False, repr=False)
    _runner: asyncio.Runner | None = field(default=None, init=False, repr=False)
    _pending_endpoint: RuntimeEndpoint | None = field(
        default=None, init=False, repr=False
    )
    _pending_instance_id: str | None = field(default=None, init=False, repr=False)

    def discovery(self) -> dict[str, object] | None:
        """Return the single canonical daemon discovery document."""

        try:
            discovery = read_daemon_discovery(self.home)
        except DaemonDiscoveryError:
            return None
        return discovery.to_document() if discovery is not None else None

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
        result: dict[str, Any] = {
            "up": up,
            "ready": False,
            "mode": "detached",
            "home": str(self.home),
            "pid": pid,
            "detail": "detached daemon running"
            if up
            else "detached daemon not running",
        }
        if not up or pid is None:
            return result
        connection = self._connection_for_pid(pid)
        if connection is None:
            result["detail"] = "detached daemon running without canonical discovery"
            return result
        discovery, observer = connection
        existing_controller = self._controller
        controller = self._controller_for_existing(pid, observer=observer)
        controller_status = self._run(controller.status())
        result["ready"] = controller_status.ready
        result["transport"] = discovery.endpoint.kind
        result["socket"] = discovery.endpoint.display
        try:
            daemon_status = self._run(observer.status())
            if isinstance(daemon_status, Success):
                result["backend"] = daemon_status.value.backend_profile
                result["url"] = daemon_status.value.ui_url
        finally:
            self._run(observer.close())
            if existing_controller is None:
                self._controller = None
        return result

    def health(self) -> dict[str, Any]:
        if self.in_process:
            return {"live": True, "mode": "in_process"}
        status = self.status()
        return {
            "live": bool(status.get("up") and status.get("ready")),
            "mode": "detached",
            **({"pid": status["pid"]} if status.get("pid") else {}),
            **({"backend": status["backend"]} if "backend" in status else {}),
        }

    def logs(self, *, follow: bool = False) -> list[str]:
        del follow
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
                metadata={"mode": "detached", **status},
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
        pid = self._recorded_pid()
        if pid is not None and _pid_alive(pid):
            raise DaemonStartError(f"daemon already running (pid={pid})")
        if not self._cleanup_runtime_records():
            raise DaemonStartError("another daemon boot owns the runtime scope")
        controller = self._new_controller(backend=backend)
        outcome = self._run(controller.start())
        if isinstance(outcome, Failure):
            self._cleanup_runtime_records()
            raise DaemonStartError(
                outcome.error.message,
                log_path=self.home / "logs" / "potpied.log",
            )
        endpoint = self._pending_endpoint
        instance_id = self._pending_instance_id
        if endpoint is None or instance_id is None or outcome.value.pid is None:
            self._run(controller.stop())
            self._cleanup_runtime_records()
            raise DaemonStartError("daemon boot identity was not retained")
        discovery = canonical_discovery(
            home=self.home,
            instance_id=instance_id,
            pid=outcome.value.pid,
            endpoint=endpoint,
        )
        try:
            write_daemon_pid(self.home, outcome.value.pid)
            write_daemon_discovery(self.home, discovery)
        except Exception:
            self._run(controller.stop())
            self._cleanup_runtime_records()
            raise
        self._controller = controller
        status = self.status()
        return {
            "pid": outcome.value.pid,
            "socket": endpoint.display,
            "bind": f"{endpoint.kind}:{endpoint.display}",
            "url": status.get("url", ""),
        }

    def stop(self) -> dict[str, Any]:
        controller = self._controller
        pid = controller.pid if controller is not None else self._recorded_pid()
        if pid is None:
            if not self._cleanup_runtime_records():
                self._raise_cleanup_ownership_conflict(expected_pid=None)
            return {"detail": "daemon not running"}
        if not _pid_alive(pid):
            if not self._cleanup_runtime_records(expected_pid=pid):
                self._raise_cleanup_ownership_conflict(expected_pid=pid)
            return {"detail": "stale pid file removed"}
        expected_instance_id = (
            controller.instance_id if controller is not None else None
        )
        if controller is None:
            connection = self._connection_for_pid(pid)
            if connection is None:
                raise DaemonStopError(
                    ResourceLifecycleError(
                        code="daemon_attached_identity_unavailable",
                        message=(
                            "refusing to signal the recorded process because its "
                            "daemon identity could not be authenticated"
                        ),
                        details={"pid": pid},
                        recommended_next_action=(
                            "inspect daemon status and runtime records before "
                            "manual recovery"
                        ),
                        retry_posture="safe",
                    )
                )
            discovery, observer = connection
            expected_instance_id = discovery.instance_id
            controller = self._controller_for_existing(pid, observer=observer)
        outcome = self._run(controller.stop())
        if isinstance(outcome, Failure):
            self._controller = None
            raise DaemonStopError(outcome.error)
        self._cleanup_runtime_records_under_lock(
            expected_instance_id=expected_instance_id,
            expected_pid=pid,
        )
        self._controller = None
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

    def _controller_for_existing(
        self,
        pid: int,
        *,
        observer: DaemonObserver,
    ) -> DaemonController:
        if self._controller is not None and self._controller.pid == pid:
            return self._controller
        controller = self._new_controller(backend=None)
        self._run(
            controller.attach(
                process=_RecordedDaemonProcess(pid),
                observer=observer,
                launch=None,
            )
        )
        self._controller = controller
        return controller

    def _connection_for_pid(self, pid: int):
        try:
            connection = load_daemon_connection(self.home)
        except DaemonDiscoveryError:
            return None
        if connection.discovery.pid != pid:
            return None
        return (
            connection.discovery,
            DaemonObserver(
                endpoint=connection.discovery.endpoint,
                bearer_token=connection.bearer_token,
                expected_instance_id=connection.discovery.instance_id,
            ),
        )

    def _run(self, awaitable: Any) -> Any:
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(awaitable)

    def _boot_spec(self, *, backend: str | None) -> DaemonBootSpec:
        instance_id = str(uuid4())
        endpoint = select_runtime_endpoint(self.home, instance_id=instance_id)
        bearer_token = generate_bearer_token()
        self._pending_endpoint = endpoint
        self._pending_instance_id = instance_id
        return DaemonBootSpec(
            launch=self._launch_spec(
                backend=backend,
                endpoint=endpoint,
                instance_id=instance_id,
                ui_port=_available_loopback_port(),
                bearer_token=bearer_token,
            ),
            observer=DaemonObserver(
                endpoint=endpoint,
                bearer_token=bearer_token,
                expected_instance_id=instance_id,
            ),
        )

    def _launch_spec(
        self,
        *,
        backend: str | None,
        endpoint: RuntimeEndpoint,
        instance_id: str,
        ui_port: int,
        bearer_token: str,
    ) -> DaemonLaunchSpec:
        from potpie.cli.telemetry.settings import load_cli_runtime_settings
        from potpie.runtime.settings import (
            project_child_environment,
        )

        overrides = {
            "CONTEXT_ENGINE_HOME": str(self.home.resolve()),
            "POTPIE_DAEMON_ENDPOINT_KIND": endpoint.kind,
            "POTPIE_DAEMON_ENDPOINT_ADDRESS": endpoint.address,
            "POTPIE_DAEMON_INSTANCE_ID": instance_id,
            "POTPIE_DAEMON_UI_PORT": str(ui_port),
            "POTPIE_DAEMON_BEARER_TOKEN": bearer_token,
            **(
                {"POTPIE_DAEMON_ENDPOINT_PORT": str(endpoint.port)}
                if endpoint.port
                else {}
            ),
            **({"CONTEXT_ENGINE_BACKEND": backend} if backend else {}),
        }
        environment = project_child_environment(
            load_cli_runtime_settings(), os.environ, overrides=overrides
        )
        return DaemonLaunchSpec(
            command=(sys.executable, "-m", "potpie.daemon"),
            environment=environment,
            log_path=self.home / "logs" / "potpied.log",
        )

    def _recorded_pid(self) -> int | None:
        return read_daemon_pid(self.home)

    def _cleanup_runtime_records(
        self,
        *,
        expected_instance_id: str | None = None,
        expected_pid: int | None = None,
    ) -> bool:
        ownership = RuntimeOwnershipLock((self.home / "daemon.runtime.lock").resolve())
        acquired = ownership.acquire()
        if isinstance(acquired, Failure):
            return False
        try:
            remove_daemon_runtime_records(
                self.home,
                expected_instance_id=expected_instance_id,
                expected_pid=expected_pid,
            )
        finally:
            ownership.release()
        return True

    def _cleanup_runtime_records_under_lock(
        self,
        *,
        expected_instance_id: str | None,
        expected_pid: int,
    ) -> None:
        ownership = RuntimeOwnershipLock((self.home / "daemon.runtime.lock").resolve())
        acquired = ownership.acquire()
        if isinstance(acquired, Failure):
            self._raise_cleanup_ownership_conflict(expected_pid=expected_pid)
        try:
            remove_daemon_runtime_records(
                self.home,
                expected_instance_id=expected_instance_id,
                expected_pid=expected_pid,
            )
        finally:
            ownership.release()

    def _raise_cleanup_ownership_conflict(self, *, expected_pid: int | None) -> None:
        raise DaemonStopError(
            ResourceLifecycleError(
                code="daemon_cleanup_ownership_conflict",
                message=(
                    "runtime records could not be cleaned without risking an "
                    "active or replacement daemon"
                ),
                details={**({"pid": expected_pid} if expected_pid else {})},
                recommended_next_action=(
                    "retry daemon status after the active boot settles"
                ),
                retry_posture="safe",
            )
        )


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


__all__ = ["Daemon", "DaemonStartError", "DaemonStopError"]
