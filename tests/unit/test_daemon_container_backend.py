import logging
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from potpie.services.managed_services.container_backend import ContainerBackend
from potpie.daemon.ports.shell import HealthStatus, ReadyProbe, ServiceSpec
from potpie.daemon.daemon_runtime.context import ServiceEndpoints, ShellContext


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("t"),
        endpoints=ServiceEndpoints(),
    )


def _spec(name="g") -> ServiceSpec:
    return ServiceSpec(
        name=name,
        backend="container",
        config={"image": "alpine:latest", "ports": {"7687": 7687}, "env": {"X": "1"}},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:7687"),
        endpoint="bolt://127.0.0.1:7687",
    )


def _http_spec(name="g") -> ServiceSpec:
    return ServiceSpec(
        name=name,
        backend="container",
        config={"image": "alpine:latest"},
        ready=ReadyProbe(kind="http", target="http://127.0.0.1:9999/health"),
        endpoint="http://127.0.0.1:9999",
    )


@pytest.mark.anyio
async def test_start_invokes_docker_run(ctx):
    be = ContainerBackend()
    fake_proc = MagicMock(returncode=0)
    fake_proc.communicate = AsyncMock(return_value=(b"abc123\n", b""))
    with patch(
        "potpie.services.managed_services.container_backend.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=fake_proc),
    ) as run:
        await be.start(_spec(), ctx)
        argv = run.await_args_list[-1].args
        assert "docker" in argv[0] or argv[0].endswith("docker")
        assert "run" in argv
        assert "-d" in argv
        assert "alpine:latest" in argv
        assert "--rm" in argv
        assert "--name" in argv
        assert "potpie_g" in argv
        # port mapping -p 7687:7687
        assert "-p" in argv
        assert "7687:7687" in argv
        # env -e X=1
        assert "-e" in argv
        assert "X=1" in argv
        # image is the final positional (no command in this spec)
        assert argv[-1] == "alpine:latest"


@pytest.mark.anyio
async def test_stop_invokes_docker_stop(ctx):
    be = ContainerBackend()
    be._containers["g"] = "abc123"
    fake_proc = MagicMock(returncode=0)
    fake_proc.communicate = AsyncMock(return_value=(b"", b""))
    with patch(
        "potpie.services.managed_services.container_backend.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=fake_proc),
    ) as run:
        await be.stop(_spec())
        argvs = [c.args for c in run.await_args_list]
        flat = [tok for argv in argvs for tok in argv]
        assert "stop" in flat
        assert "abc123" in flat
        assert "rm" in flat


@pytest.mark.anyio
async def test_start_failure_raises(ctx):
    be = ContainerBackend()
    fake_proc = MagicMock(returncode=125)
    fake_proc.communicate = AsyncMock(return_value=(b"", b"image not found"))
    with patch(
        "potpie.services.managed_services.container_backend.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=fake_proc),
    ):
        with pytest.raises(RuntimeError) as ei:
            await be.start(_spec(), ctx)
        assert "image not found" in str(ei.value)


@pytest.mark.anyio
async def test_probe_http_kind_returns_ready(ctx):
    be = ContainerBackend()
    with patch(
        "potpie.services.managed_services.container_backend._http_probe",
        new=AsyncMock(return_value=True),
    ):
        assert await be.probe(_http_spec()) is HealthStatus.READY
