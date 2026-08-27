from __future__ import annotations

# ruff: noqa: S101 - pytest integration tests use assertions intentionally.

import signal
import stat
import sys
from pathlib import Path

import pytest

from potpie.runtime import (
    DaemonController,
    DaemonBootSpec,
    DaemonLaunchSpec,
    ProtocolTransportError,
    RuntimeEndpoint,
    StopResult,
    DaemonObserver,
    generate_bearer_token,
)
from potpie_context_engine import Failure, Success


_CHILD_RUNTIME = """
import asyncio
import os
from pathlib import Path
from potpie.runtime import CanonicalDaemonRuntime, RuntimeEndpoint, run_foreground
from potpie_context_engine import Success

class Handler:
    async def handle(self, request, *, authentication):
        return Success({"operation": request.operation.value})

runtime = CanonicalDaemonRuntime(
    endpoint=RuntimeEndpoint(kind="uds", address=os.environ["TEST_DAEMON_SOCKET"]),
    bearer_token=os.environ["TEST_DAEMON_TOKEN"],
    operation_handler=Handler(),
    ownership_lock_path=Path(os.environ["TEST_DAEMON_LOCK"]),
    instance_id=os.environ["TEST_DAEMON_INSTANCE"],
)
asyncio.run(run_foreground(runtime))
"""


class _FallbackObserver:
    def __init__(self, *, ready: bool) -> None:
        self._ready = ready
        self.closed = 0

    @property
    def instance_id(self) -> str:
        return "fallback-instance"

    async def wait_ready(self, process, *, timeout_s: float):
        del process, timeout_s
        if self._ready:
            return Success(None)
        return Failure(
            ProtocolTransportError(
                code="not_ready",
                message="not ready",
                retry_posture="safe",
            )
        )

    async def request_stop(self):
        return Failure(
            ProtocolTransportError(
                code="unresponsive",
                message="unresponsive",
                retry_posture="unknown",
            )
        )

    async def close(self) -> None:
        self.closed += 1


class _AttachedProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    async def wait(self) -> int:
        self.wait_calls += 1
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


@pytest.mark.anyio
async def test_controller_starts_observes_and_typed_stops_real_foreground_child(
    tmp_path: Path,
    short_socket_dir: Path,
) -> None:
    socket_path = short_socket_dir / "controller.sock"
    log_path = tmp_path / "daemon.log"
    log_path.write_text("controller boot\n", encoding="utf-8")
    token = generate_bearer_token()
    instance_id = "controller-instance"
    endpoint = RuntimeEndpoint(kind="uds", address=str(socket_path))
    observer = DaemonObserver(
        endpoint=endpoint,
        bearer_token=token,
        expected_instance_id=instance_id,
    )
    controller = DaemonController(
        boot_factory=lambda: DaemonBootSpec(
            launch=DaemonLaunchSpec(
                command=(sys.executable, "-c", _CHILD_RUNTIME),
                cwd=tmp_path,
                environment={
                    "TEST_DAEMON_SOCKET": str(socket_path),
                    "TEST_DAEMON_TOKEN": token,
                    "TEST_DAEMON_INSTANCE": instance_id,
                    "TEST_DAEMON_LOCK": str(short_socket_dir / "controller.lock"),
                },
                log_path=log_path,
            ),
            observer=observer,
        ),
        readiness_timeout_s=5,
        stop_timeout_s=2,
    )

    started = await controller.start()
    status = await controller.status()
    stopped = await controller.stop()

    assert isinstance(started, Success)
    assert started.value.running is True
    assert started.value.ready is True
    assert started.value.pid is not None
    assert status.running is True
    assert status.ready is True
    assert status.instance_id == instance_id
    assert controller.logs() == ["controller boot"]
    assert isinstance(stopped, Success)
    assert stopped.value.mode == "typed_shutdown"
    assert controller.pid is None
    assert not socket_path.exists()
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


@pytest.mark.anyio
async def test_controller_falls_back_to_bounded_signal_termination() -> None:
    observer = _FallbackObserver(ready=True)
    controller = DaemonController(
        boot_factory=lambda: DaemonBootSpec(
            launch=DaemonLaunchSpec(
                command=(sys.executable, "-c", "import time; time.sleep(60)"),
            ),
            observer=observer,
        ),
        readiness_timeout_s=1,
        stop_timeout_s=0.5,
    )

    started = await controller.start()
    stopped = await controller.stop()

    assert isinstance(started, Success)
    assert isinstance(stopped, Success)
    assert stopped.value == StopResult(
        mode="terminated", exit_code=-int(signal.SIGTERM)
    )
    assert observer.closed == 1
    assert controller.pid is None


@pytest.mark.anyio
@pytest.mark.parametrize("ready", (False, True))
async def test_attached_controller_refuses_signal_fallback(ready: bool) -> None:
    process = _AttachedProcess()
    observer = _FallbackObserver(ready=ready)
    controller = DaemonController(
        boot_factory=lambda: pytest.fail("attached stop must not compose a boot"),
        readiness_timeout_s=1,
        stop_timeout_s=0.1,
    )

    attached = await controller.attach(process=process, observer=observer)
    stopped = await controller.stop()

    assert isinstance(attached, Success)
    assert attached.value.ready is ready
    assert isinstance(stopped, Failure)
    assert stopped.error.code == "daemon_attached_shutdown_unavailable"
    assert stopped.error.retry_posture == "safe"
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.wait_calls == 0
    assert observer.closed == 1


@pytest.mark.anyio
async def test_controller_restart_composes_fresh_boot_identity(
    tmp_path: Path,
    short_socket_dir: Path,
) -> None:
    boot_count = 0

    def boot_factory() -> DaemonBootSpec:
        nonlocal boot_count
        boot_count += 1
        token = generate_bearer_token()
        instance_id = f"restart-instance-{boot_count}"
        endpoint = RuntimeEndpoint(
            kind="uds",
            address=str(short_socket_dir / f"restart-{boot_count}.sock"),
        )
        return DaemonBootSpec(
            launch=DaemonLaunchSpec(
                command=(sys.executable, "-c", _CHILD_RUNTIME),
                cwd=tmp_path,
                environment={
                    "TEST_DAEMON_SOCKET": endpoint.address,
                    "TEST_DAEMON_TOKEN": token,
                    "TEST_DAEMON_INSTANCE": instance_id,
                    "TEST_DAEMON_LOCK": str(
                        short_socket_dir / f"restart-{boot_count}.lock"
                    ),
                },
            ),
            observer=DaemonObserver(
                endpoint=endpoint,
                bearer_token=token,
                expected_instance_id=instance_id,
            ),
        )

    controller = DaemonController(
        boot_factory=boot_factory,
        readiness_timeout_s=5,
        stop_timeout_s=2,
    )
    first = await controller.start()
    assert isinstance(first, Success)

    restarted = await controller.restart()

    assert isinstance(restarted, Success)
    assert restarted.value.instance_id == "restart-instance-2"
    assert restarted.value.instance_id != first.value.instance_id
    assert boot_count == 2
    assert isinstance(await controller.stop(), Success)


@pytest.mark.anyio
async def test_readiness_failure_terminates_owned_child_and_returns_typed_error() -> (
    None
):
    observer = _FallbackObserver(ready=False)
    controller = DaemonController(
        boot_factory=lambda: DaemonBootSpec(
            launch=DaemonLaunchSpec(
                command=(sys.executable, "-c", "import time; time.sleep(60)"),
            ),
            observer=observer,
        ),
        readiness_timeout_s=0.1,
        stop_timeout_s=0.5,
    )

    outcome = await controller.start()

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "daemon_readiness_failed"
    assert controller.pid is None
    assert observer.closed == 1
