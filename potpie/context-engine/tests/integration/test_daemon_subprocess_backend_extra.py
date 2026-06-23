"""Additional subprocess backend coverage: http probe, cmd probe, degraded path."""

from __future__ import annotations

import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from potpie.context_engine.adapters.outbound.managed_services.subprocess_backend import SubprocessBackend
from potpie.context_engine.domain.ports.daemon.shell import HealthStatus, ReadyProbe, ServiceSpec
from potpie.context_engine.host.daemon_runtime.context import ShellContext


@pytest.mark.anyio
async def test_probe_returns_stopped_when_no_proc():
    be = SubprocessBackend()
    spec = ServiceSpec(
        name="missing",
        backend="subprocess",
        config={"command": ["python", "-c", "pass"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    assert await be.probe(spec) is HealthStatus.STOPPED


@pytest.mark.anyio
async def test_probe_returns_degraded_after_exit(
    daemon_ctx: ShellContext, tmp_path: pathlib.Path
):
    spec = ServiceSpec(
        name="exit0",
        backend="subprocess",
        config={"command": ["python", "-c", "import sys; sys.exit(0)"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    # Wait for process to exit
    for _ in range(50):
        proc = be._procs.get("exit0")
        if proc and proc.returncode is not None:
            break
        await asyncio.sleep(0.05)
    assert await be.probe(spec) is HealthStatus.DEGRADED
    await be.stop(spec)


@pytest.mark.anyio
async def test_probe_http_kind_returns_starting_on_no_server(
    daemon_ctx: ShellContext, tmp_path: pathlib.Path
):
    """http probe kind returns STARTING when endpoint is unreachable."""
    spec = ServiceSpec(
        name="httptest",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(
            kind="http", target="http://127.0.0.1:1/health", interval_s=0.1
        ),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.STARTING
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_cmd_kind_returns_ready_on_success(
    daemon_ctx: ShellContext, tmp_path: pathlib.Path
):
    """cmd probe kind returns READY when command exits 0."""
    spec = ServiceSpec(
        name="cmdtest",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(
            kind="cmd", target="python -c 'import sys; sys.exit(0)'", interval_s=2.0
        ),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.READY
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_cmd_kind_returns_starting_on_failure(
    daemon_ctx: ShellContext, tmp_path: pathlib.Path
):
    """cmd probe kind returns STARTING when command exits non-zero."""
    spec = ServiceSpec(
        name="cmdfail",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(
            kind="cmd", target="python -c 'import sys; sys.exit(1)'", interval_s=2.0
        ),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.STARTING
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_unknown_kind_returns_degraded(
    daemon_ctx: ShellContext, tmp_path: pathlib.Path
):
    """Unknown probe kind returns DEGRADED."""
    spec = ServiceSpec(
        name="unknown",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="unknown", target="whatever"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.DEGRADED
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_tcp_failure_returns_starting(daemon_ctx: ShellContext):
    spec = ServiceSpec(
        name="tcpfail",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1", interval_s=0.1),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    try:
        assert await be.probe(spec) is HealthStatus.STARTING
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_stop_already_exited_proc(
    daemon_ctx: ShellContext, tmp_path: pathlib.Path
):
    """Stopping an already-exited process must not raise."""
    spec = ServiceSpec(
        name="quickexit",
        backend="subprocess",
        config={"command": ["python", "-c", "pass"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    # Let it exit naturally
    for _ in range(50):
        proc = be._procs.get("quickexit")
        if proc and proc.returncode is not None:
            break
        await asyncio.sleep(0.05)
    # Stop on already-exited process should be safe
    await be.stop(spec)


@pytest.mark.anyio
async def test_start_closes_parent_log_file_after_spawn(
    daemon_ctx: ShellContext, monkeypatch
):
    spec = ServiceSpec(
        name="fdtest",
        backend="subprocess",
        config={"command": ["python", "-c", "pass"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    fake_proc = MagicMock(returncode=None)
    captured = {}

    async def fake_create_subprocess_exec(*args, stdout, **kwargs):
        captured["stdout"] = stdout
        return fake_proc

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.managed_services.subprocess_backend.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=fake_create_subprocess_exec),
    )

    await be.start(spec, daemon_ctx)

    assert captured["stdout"].closed
