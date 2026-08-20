"""External direct-child controller for one foreground daemon process."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from potpie.runtime.clients import DaemonControlClient
from potpie.runtime.protocol import RuntimeBoundaryError
from potpie.runtime.resource_manager import ResourceLifecycleError
from potpie.runtime.transport import HttpDaemonTransport, RuntimeEndpoint
from potpie_context_engine import Failure, Success


@dataclass(frozen=True, slots=True)
class DaemonLaunchSpec:
    command: tuple[str, ...]
    cwd: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    log_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.command or not self.command[0].strip():
            raise ValueError("daemon launch command must not be empty")


@dataclass(frozen=True, slots=True)
class ControllerStatus:
    running: bool
    ready: bool
    pid: int | None
    instance_id: str | None
    exit_code: int | None


@dataclass(frozen=True, slots=True)
class StopResult:
    mode: Literal["already_stopped", "typed_shutdown", "terminated", "killed"]
    exit_code: int | None


ControllerStartOutcome: TypeAlias = (
    Success[ControllerStatus] | Failure[ResourceLifecycleError]
)
ControllerStopOutcome: TypeAlias = Success[StopResult] | Failure[ResourceLifecycleError]


class DaemonProcessObserver(Protocol):
    @property
    def instance_id(self) -> str | None: ...

    async def wait_ready(
        self, process: DaemonProcessHandle, *, timeout_s: float
    ) -> Success[None] | Failure[RuntimeBoundaryError]: ...

    async def request_stop(
        self,
    ) -> Success[object] | Failure[RuntimeBoundaryError]: ...

    async def close(self) -> None: ...


class DaemonProcessHandle(Protocol):
    """Minimal process surface used for owned and previously launched children."""

    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int | None: ...

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DaemonBootSpec:
    """Fresh launch and observer pair for exactly one daemon boot."""

    launch: DaemonLaunchSpec
    observer: DaemonProcessObserver


DaemonBootFactory: TypeAlias = Callable[[], DaemonBootSpec]


class TypedDaemonObserver:
    """Observe readiness and stop through the authenticated typed protocol."""

    def __init__(
        self,
        *,
        endpoint: RuntimeEndpoint,
        bearer_token: str,
        expected_instance_id: str,
        poll_interval_s: float = 0.05,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("readiness poll interval must be positive")
        self._transport = HttpDaemonTransport(
            endpoint=endpoint,
            bearer_token=bearer_token,
        )
        self._client = DaemonControlClient(
            transport=self._transport,
            expected_instance_id=expected_instance_id,
        )
        self._instance_id = expected_instance_id
        self._poll_interval_s = poll_interval_s

    @property
    def instance_id(self) -> str:
        return self._instance_id

    async def wait_ready(
        self, process: DaemonProcessHandle, *, timeout_s: float
    ) -> Success[None] | Failure[RuntimeBoundaryError]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_failure: Failure[RuntimeBoundaryError] | None = None
        while loop.time() < deadline:
            if process.returncode is not None:
                break
            handshake = await self._client.handshake()
            if isinstance(handshake, Success):
                return Success(None)
            last_failure = handshake
            await asyncio.sleep(
                min(self._poll_interval_s, max(0.0, deadline - loop.time()))
            )
        if last_failure is not None:
            return last_failure
        return Failure(
            ResourceLifecycleError(
                code="daemon_exited_before_ready",
                message="daemon process exited before readiness",
                details={"exit_code": process.returncode},
            )
        )

    async def request_stop(
        self,
    ) -> Success[object] | Failure[RuntimeBoundaryError]:
        if self._client.handshake_result is None:
            handshake = await self._client.handshake()
            if isinstance(handshake, Failure):
                return handshake
        return await self._client.shutdown(reason="controller_requested")

    async def close(self) -> None:
        await self._transport.close()


class DaemonController:
    """Create, observe, stop, and clean up one directly owned child process."""

    def __init__(
        self,
        *,
        boot_factory: DaemonBootFactory,
        readiness_timeout_s: float = 10.0,
        stop_timeout_s: float = 5.0,
        log_paths: tuple[Path, ...] = (),
    ) -> None:
        if readiness_timeout_s <= 0 or stop_timeout_s <= 0:
            raise ValueError("controller timeouts must be positive")
        self._boot_factory = boot_factory
        self._launch: DaemonLaunchSpec | None = None
        self._observer: DaemonProcessObserver | None = None
        self._readiness_timeout_s = readiness_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._process: DaemonProcessHandle | None = None
        self._log_handle: object | None = None
        self._log_paths = log_paths
        self._lock = asyncio.Lock()

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.returncode is None else None

    async def start(self) -> ControllerStartOutcome:
        async with self._lock:
            if self._process is not None and self._process.returncode is None:
                return Failure(
                    ResourceLifecycleError(
                        code="daemon_already_running",
                        message="the controller already owns a running daemon",
                    )
                )
            try:
                boot = self._boot_factory()
            except Exception as exc:
                return Failure(
                    ResourceLifecycleError(
                        code="daemon_boot_composition_failed",
                        message="the controller could not compose a fresh daemon boot",
                        details={"error_type": type(exc).__name__},
                    )
                )
            self._launch = boot.launch
            self._observer = boot.observer
            try:
                log_target = await self._open_log_target()
            except OSError as exc:
                await self._close_observer(boot.observer)
                return Failure(
                    ResourceLifecycleError(
                        code="daemon_log_open_failed",
                        message="the controller could not open the daemon log",
                        details={"error_type": type(exc).__name__},
                    )
                )
            environment = dict(os.environ)
            environment.update(boot.launch.environment)
            try:
                process = await asyncio.create_subprocess_exec(
                    *boot.launch.command,
                    cwd=boot.launch.cwd,
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log_target,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except (OSError, ValueError) as exc:
                await self._close_log_target()
                return Failure(
                    ResourceLifecycleError(
                        code="daemon_process_creation_failed",
                        message="the controller could not create the daemon process",
                        details={"error_type": type(exc).__name__},
                        recommended_next_action="inspect the configured daemon command",
                    )
                )
            self._process = process
            try:
                ready = await boot.observer.wait_ready(
                    process,
                    timeout_s=self._readiness_timeout_s,
                )
            except Exception as exc:
                ready = Failure(
                    ResourceLifecycleError(
                        code="daemon_observer_failed",
                        message="the daemon readiness observer failed",
                        details={"error_type": type(exc).__name__},
                    )
                )
            if isinstance(ready, Failure):
                await self._bounded_terminate(process)
                await self._close_observer(boot.observer)
                await self._close_log_target()
                return Failure(
                    ResourceLifecycleError(
                        code="daemon_readiness_failed",
                        message="the daemon did not complete an authenticated handshake",
                        details={
                            "cause_category": ready.error.category,
                            "cause_code": ready.error.code,
                        },
                        recommended_next_action="inspect daemon logs",
                        retry_posture="safe",
                    )
                )
            return Success(self._status(ready=True))

    async def attach(
        self,
        *,
        process: DaemonProcessHandle,
        observer: DaemonProcessObserver,
        launch: DaemonLaunchSpec | None = None,
    ) -> ControllerStartOutcome:
        """Observe a daemon launched by an earlier controller invocation.

        Local CLI commands are short-lived, while the foreground daemon child is
        intentionally long-lived. A later invocation can therefore reattach to
        the exact PID and discovery identity recorded by the earlier invocation;
        this is not external-supervisor integration.
        """
        async with self._lock:
            current = self._process
            if (
                current is not None
                and current.returncode is None
                and current.pid != process.pid
            ):
                return Failure(
                    ResourceLifecycleError(
                        code="daemon_controller_busy",
                        message="the controller already observes another daemon",
                    )
                )
            self._process = process
            self._observer = observer
            self._launch = launch
            ready = await observer.wait_ready(process, timeout_s=0.25)
            return Success(self._status(ready=isinstance(ready, Success)))

    async def status(self) -> ControllerStatus:
        process = self._process
        if process is None or process.returncode is not None:
            return self._status(ready=False)
        observer = self._observer
        if observer is None:
            return self._status(ready=False)
        ready = await observer.wait_ready(process, timeout_s=0.25)
        return self._status(ready=isinstance(ready, Success))

    async def stop(self) -> ControllerStopOutcome:
        async with self._lock:
            process = self._process
            observer = self._observer
            if process is None or process.returncode is not None:
                if observer is not None:
                    await self._close_observer(observer)
                await self._close_log_target()
                return Success(
                    StopResult(
                        mode="already_stopped",
                        exit_code=process.returncode if process is not None else None,
                    )
                )
            if observer is None:
                requested = Failure(
                    ResourceLifecycleError(
                        code="daemon_observer_missing",
                        message="the owned daemon has no process observer",
                    )
                )
            else:
                try:
                    requested = await observer.request_stop()
                except Exception as exc:
                    requested = Failure(
                        ResourceLifecycleError(
                            code="daemon_stop_observer_failed",
                            message="the daemon stop observer failed",
                            details={"error_type": type(exc).__name__},
                        )
                    )
            if isinstance(requested, Success):
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self._stop_timeout_s
                    )
                    if process.returncode == 0:
                        result = StopResult(
                            mode="typed_shutdown", exit_code=process.returncode
                        )
                    else:
                        await self._close_observer(observer)
                        await self._close_log_target()
                        return Failure(
                            ResourceLifecycleError(
                                code="daemon_shutdown_failed",
                                message="daemon exited unsuccessfully after typed shutdown",
                                details={"exit_code": process.returncode},
                                recommended_next_action="inspect daemon logs",
                            )
                        )
                except TimeoutError:
                    result = await self._bounded_terminate(process)
            else:
                result = await self._bounded_terminate(process)
            if observer is not None:
                await self._close_observer(observer)
            await self._close_log_target()
            return Success(result)

    async def restart(self) -> ControllerStartOutcome:
        stopped = await self.stop()
        if isinstance(stopped, Failure):
            return stopped
        return await self.start()

    def logs(self) -> list[str]:
        """Read the configured daemon logs without changing process state."""
        candidates: list[Path] = []
        if self._launch is not None and self._launch.log_path is not None:
            candidates.append(self._launch.log_path)
        candidates.extend(path for path in self._log_paths if path not in candidates)
        for path in candidates:
            if path.exists():
                return path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
        return []

    async def _bounded_terminate(
        self, process: DaemonProcessHandle
    ) -> StopResult:
        if process.returncode is not None:
            return StopResult(mode="already_stopped", exit_code=process.returncode)
        try:
            process.terminate()
        except ProcessLookupError:
            await process.wait()
            return StopResult(mode="already_stopped", exit_code=process.returncode)
        try:
            await asyncio.wait_for(process.wait(), timeout=self._stop_timeout_s)
            return StopResult(mode="terminated", exit_code=process.returncode)
        except TimeoutError:
            process.kill()
            await process.wait()
            return StopResult(mode="killed", exit_code=process.returncode)

    def _status(self, *, ready: bool) -> ControllerStatus:
        process = self._process
        running = process is not None and process.returncode is None
        observer = self._observer
        return ControllerStatus(
            running=running,
            ready=ready and running,
            pid=process.pid if running else None,
            instance_id=(
                observer.instance_id if running and observer is not None else None
            ),
            exit_code=process.returncode if process is not None else None,
        )

    async def _open_log_target(self) -> object:
        launch = self._launch
        if launch is None or launch.log_path is None:
            return asyncio.subprocess.DEVNULL
        path = launch.log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(  # noqa: PTH123 - explicit owner-only file mode
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        os.chmod(path, 0o600)
        self._log_handle = os.fdopen(descriptor, "ab")  # noqa: PTH123
        return self._log_handle

    async def _close_log_target(self) -> None:
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()  # type: ignore[attr-defined]

    @staticmethod
    async def _close_observer(observer: DaemonProcessObserver) -> None:
        with contextlib.suppress(Exception):
            await observer.close()


__all__ = [
    "ControllerStartOutcome",
    "ControllerStatus",
    "ControllerStopOutcome",
    "DaemonBootFactory",
    "DaemonBootSpec",
    "DaemonController",
    "DaemonLaunchSpec",
    "DaemonProcessHandle",
    "DaemonProcessObserver",
    "StopResult",
    "TypedDaemonObserver",
]
