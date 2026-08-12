import asyncio
import pathlib
import sys

import pytest

from potpie.daemon.managed_services.subprocess_backend import SubprocessBackend
from potpie.daemon.ports.shell import HealthStatus, ReadyProbe, ServiceSpec
from potpie.daemon.runtime.context import ShellContext
from tests.conftest import free_port

#: The interpreter to spawn stub services with. Never the bare name ``python``:
#: that is a PATH lookup, and a machine whose only interpreter is ``python3`` —
#: or a virtualenv invoked by absolute path — has no such entry, so every test
#: here died in ``Popen`` with ``FileNotFoundError: 'python'`` before the
#: backend under test had done anything. ``sys.executable`` exists wherever the
#: suite is running.
PYTHON = sys.executable


@pytest.mark.anyio
async def test_start_and_probe_tcp(tmp_path: pathlib.Path, daemon_ctx: ShellContext):
    port = free_port()
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
        config={"command": [PYTHON, str(script), str(port)]},
        ready=ReadyProbe(kind="tcp", target=f"127.0.0.1:{port}"),
        endpoint=f"tcp://127.0.0.1:{port}",
    )
    be = SubprocessBackend()
    try:
        await be.start(spec, daemon_ctx)
        for _ in range(60):
            if await be.probe(spec) is HealthStatus.READY:
                break
            await asyncio.sleep(0.1)
        assert await be.probe(spec) is HealthStatus.READY
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_probe_returns_starting_before_ready(
    daemon_ctx: ShellContext, tmp_path: pathlib.Path
):
    spec = ServiceSpec(
        name="noop",
        backend="subprocess",
        config={"command": [PYTHON, "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),  # unreachable
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    try:
        await be.start(spec, daemon_ctx)
        assert await be.probe(spec) is HealthStatus.STARTING
    finally:
        await be.stop(spec)


@pytest.mark.anyio
async def test_stop_terminates(daemon_ctx: ShellContext, tmp_path: pathlib.Path):
    spec = ServiceSpec(
        name="x",
        backend="subprocess",
        config={"command": [PYTHON, "-c", "import time; time.sleep(60)"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    be = SubprocessBackend()
    await be.start(spec, daemon_ctx)
    await be.stop(spec)
    await be.stop(spec)  # double-stop must be safe
