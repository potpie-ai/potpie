"""Canonical local daemon lifecycle controlled by Potpie."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from potpie.daemon.discovery import (
    DaemonDiscoveryError,
    canonical_discovery,
    discovery_path,
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

    def __init__(
        self,
        message: str,
        *,
        log_path: Path | None = None,
        recommended_next_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.log_path = log_path
        self.recommended_next_action = recommended_next_action


class DaemonStopError(Exception):
    """Raised when a detached daemon cannot be stopped safely."""

    def __init__(self, error: ResourceLifecycleError) -> None:
        super().__init__(error.message)
        self.error = error


# Command-line fragments that identify a process as one of our own daemons.
# The recorded PID alone is not proof of identity (PIDs are reused), so a
# forced takedown of an unauthenticatable daemon checks this first.
_DAEMON_COMMAND_MARKERS = ("potpie.daemon", "potpie-daemon", "potpied")


def _process_command_line(pid: int) -> str | None:
    """Best-effort command line for ``pid``; ``None`` when it cannot be read."""

    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def _looks_like_potpie_daemon(pid: int) -> bool:
    command = _process_command_line(pid)
    if command is None:
        return False
    return any(marker in command for marker in _DAEMON_COMMAND_MARKERS)


def _terminate_pid(pid: int, *, grace_s: float = 10.0) -> str:
    """SIGTERM then SIGKILL the recorded process; return how it ended."""

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_stopped"
    except OSError as exc:
        raise DaemonStopError(
            ResourceLifecycleError(
                code="daemon_signal_failed",
                message=f"could not signal the recorded daemon process: {exc}",
                details={"pid": pid},
                recommended_next_action=f"stop it manually with 'kill {pid}'",
                retry_posture="safe",
            )
        ) from exc
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return "terminated"
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    except OSError:
        return "terminated"
    return "killed"


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
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM means the process exists but this user cannot signal it. Treat
        # that as live so status/start cannot mistake a cross-user process for
        # a stale PID and attempt runtime-record cleanup.
        return True
    except OSError:
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
            result["identity"] = "absent"
            return result
        connection = self._connection_for_pid(pid)
        if connection is None:
            reason = self._discovery_failure()
            result["identity"] = "unauthenticated"
            result["detail"] = (
                "detached daemon running, but its runtime record cannot be "
                f"authenticated by this build ({reason.message})"
                if reason is not None
                else (
                    "detached daemon running, but its runtime record does not "
                    "match the recorded process"
                )
            )
            result["identity_reason"] = (
                reason.code if reason is not None else "daemon_discovery_mismatch"
            )
            result["recovery"] = (
                "run 'potpie daemon restart' to replace it "
                f"(manual escape hatch: kill {pid})"
            )
            return result
        result["identity"] = "ok"
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
            # "already running" is unhelpful when that daemon is the reason
            # every command is failing; name the operation that replaces it.
            if self._connection_for_pid(pid) is None:
                raise DaemonStartError(
                    f"daemon already running (pid={pid}), but its runtime record "
                    "cannot be authenticated by this build",
                    recommended_next_action=(
                        "run 'potpie daemon restart' to replace it "
                        f"(manual escape hatch: kill {pid})"
                    ),
                )
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

    def stop(self, *, force: bool = False) -> dict[str, Any]:
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
                if force:
                    return self._force_stop(pid)
                raise DaemonStopError(self._unauthenticated_identity_error(pid))
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
        unauthenticated = current_status.get("identity") == "unauthenticated"
        if (
            current_status.get("up")
            and not isinstance(current_backend, str)
            and not unauthenticated
        ):
            raise DaemonStopError(
                ResourceLifecycleError(
                    code="daemon_backend_undetermined",
                    message=(
                        "cannot determine the running daemon's backend; refusing "
                        "restart to avoid backend drift"
                    ),
                    details={"pid": current_status.get("pid")},
                    recommended_next_action=(
                        "check 'potpie daemon status', then stop it with "
                        "'potpie daemon stop' and start it again"
                    ),
                    retry_posture="safe",
                )
            )
        # An unauthenticatable record (a pre-upgrade daemon, a truncated or
        # legacy discovery file) has no readable backend and no typed shutdown
        # path. Restart is exactly the operation that replaces that process, so
        # take it down by signal rather than dead-ending the only recovery the
        # CLI recommends.
        if unauthenticated:
            self.stop(force=True)
        else:
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

    def _discovery_failure(self) -> DaemonDiscoveryError | None:
        """Return why the runtime record is unusable, if that is the reason."""

        try:
            read_daemon_discovery(self.home)
        except DaemonDiscoveryError as exc:
            return exc
        return None

    def _unauthenticated_identity_error(self, pid: int) -> ResourceLifecycleError:
        reason = self._discovery_failure()
        detail = f" ({reason.message})" if reason is not None else ""
        return ResourceLifecycleError(
            code="daemon_attached_identity_unavailable",
            message=(
                "refusing to signal the recorded process because its daemon "
                f"identity could not be authenticated{detail}"
            ),
            details={
                "pid": pid,
                "discovery_record": str(discovery_path(self.home)),
                **({"reason": reason.code} if reason is not None else {}),
            },
            recommended_next_action=(
                "run 'potpie daemon restart' to replace it, or "
                f"'potpie daemon stop --force'; manual escape hatch: kill {pid}"
            ),
            retry_posture="safe",
        )

    def _force_stop(self, pid: int) -> dict[str, Any]:
        """Replace a daemon whose runtime record cannot be authenticated.

        The recorded PID alone is not proof of identity, so verify the process
        still looks like one of our daemons before signalling it.
        """

        if not _looks_like_potpie_daemon(pid):
            raise DaemonStopError(
                ResourceLifecycleError(
                    code="daemon_recorded_pid_not_a_daemon",
                    message=(
                        f"the recorded process (pid={pid}) is not a potpie daemon; "
                        "refusing to signal it"
                    ),
                    details={
                        "pid": pid,
                        "discovery_record": str(discovery_path(self.home)),
                        "command": _process_command_line(pid),
                    },
                    recommended_next_action=(
                        "the runtime record is stale — remove "
                        f"{self.home} runtime files or stop pid {pid} yourself"
                    ),
                    retry_posture="safe",
                )
            )
        mode = _terminate_pid(pid)
        self._cleanup_runtime_records()
        self._controller = None
        return {
            "detail": {
                "already_stopped": "daemon not running",
                "terminated": "daemon stopped (forced: identity unauthenticated)",
                "killed": "daemon killed (forced after timeout)",
            }[mode],
            "forced": True,
            "pid": pid,
        }

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
