import asyncio
import logging
import pathlib
import socket
import pytest
from adapters.outbound.managed_services.subprocess_backend import SubprocessBackend
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
async def test_start_and_probe_tcp(tmp_path: pathlib.Path, ctx):
    port = _free_port()
    script = tmp_path / "srv.py"
    script.write_text(
        "import socket, sys\n"
        "s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen()\n"
        "while True:\n"
        "    c, _ = s.accept(); c.close()\n"
    )
    spec = ServiceSpec(
        name="stub",
        backend="subprocess",
        config={"command": ["python", str(script), str(port)]},
        ready=ReadyProbe(kind="tcp", target=f"127.0.0.1:{port}"),
        endpoint=f"tcp://127.0.0.1:{port}",
    )
    be = SubprocessBackend()
    try:
        await be.start(spec, ctx)
        for _ in range(60):
            if await be.probe(spec) is HealthStatus.READY:
                break
            await asyncio.sleep(0.1)
        assert await be.probe(spec) is HealthStatus.READY
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_returns_starting_before_ready(ctx, tmp_path: pathlib.Path):
    spec = ServiceSpec(
        name="noop",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),  # unreachable
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    try:
        await be.start(spec, ctx)
        assert await be.probe(spec) is HealthStatus.STARTING
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_stop_terminates(ctx, tmp_path: pathlib.Path):
    spec = ServiceSpec(
        name="x",
        backend="subprocess",
        config={"command": ["python", "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, ctx)
    await be.stop(spec)
    await be.stop(spec)  # double-stop must be safe
