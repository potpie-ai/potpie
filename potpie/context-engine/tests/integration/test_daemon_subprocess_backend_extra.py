"""Additional subprocess backend coverage: http probe, cmd probe, degraded path."""

from __future__ import annotations
import asyncio
import logging
import pathlib
import socket
import pytest
from adapters.outbound.managed_services.subprocess_backend import (
    SubprocessBackend,
    _tcp_probe,
)
from domain.ports.daemon.shell import ServiceSpec, ReadyProbe, HealthStatus
from host.daemon_runtime.context import ShellContext, ServiceEndpoints


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("t"),
        endpoints=ServiceEndpoints(),
    )


@pytest.mark.anyio
async def test_probe_returns_stopped_when_no_proc(ctx):
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
async def test_probe_returns_degraded_after_exit(ctx, tmp_path: pathlib.Path):
    spec = ServiceSpec(
        name="exit0",
        backend="subprocess",
        config={"command": ["python", "-c", "import sys; sys.exit(0)"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, ctx)
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
    ctx, tmp_path: pathlib.Path
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
    await be.start(spec, ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.STARTING
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_cmd_kind_returns_ready_on_success(ctx, tmp_path: pathlib.Path):
    """cmd probe kind returns READY when command exits 0."""
    spec = ServiceSpec(
        name="cmdtest",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="cmd", target="python -c pass", interval_s=2.0),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.READY
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_cmd_kind_returns_starting_on_failure(ctx, tmp_path: pathlib.Path):
    """cmd probe kind returns STARTING when command exits non-zero."""
    spec = ServiceSpec(
        name="cmdfail",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(
            kind="cmd", target="python -c import sys; sys.exit(1)", interval_s=2.0
        ),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.STARTING
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_unknown_kind_returns_degraded(ctx, tmp_path: pathlib.Path):
    """Unknown probe kind returns DEGRADED."""
    spec = ServiceSpec(
        name="unknown",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="unknown", target="whatever"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, ctx)
    try:
        result = await be.probe(spec)
        assert result is HealthStatus.DEGRADED
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_tcp_probe_failure_returns_false():
    """_tcp_probe returns False when connection refused."""
    result = await _tcp_probe("127.0.0.1", 1, timeout_s=0.1)
    assert result is False


@pytest.mark.anyio
async def test_stop_already_exited_proc(ctx, tmp_path: pathlib.Path):
    """Stopping an already-exited process must not raise."""
    spec = ServiceSpec(
        name="quickexit",
        backend="subprocess",
        config={"command": ["python", "-c", "pass"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, ctx)
    # Let it exit naturally
    for _ in range(50):
        proc = be._procs.get("quickexit")
        if proc and proc.returncode is not None:
            break
        await asyncio.sleep(0.05)
    # Stop on already-exited process should be safe
    await be.stop(spec)
