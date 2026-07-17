import logging
import pathlib
import socket
from unittest.mock import AsyncMock, patch

import pytest

from potpie_context_engine.adapters.outbound.managed_services.external_backend import ExternalBackend
from potpie_context_engine.domain.ports.daemon.shell import HealthStatus, ReadyProbe, ServiceSpec
from potpie_context_engine.host.daemon_runtime.context import ServiceEndpoints, ShellContext


def _free_port_and_listener() -> tuple[int, socket.socket]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s.getsockname()[1], s


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("t"),
        endpoints=ServiceEndpoints(),
    )


@pytest.mark.anyio
async def test_external_does_not_start_anything(ctx):
    be = ExternalBackend()
    spec = ServiceSpec(
        name="x",
        backend="external",
        config={},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    await be.start(spec, ctx)
    await be.stop(spec)


@pytest.mark.anyio
async def test_external_probe_ready_when_target_listening(ctx):
    port, sock = _free_port_and_listener()
    try:
        be = ExternalBackend()
        spec = ServiceSpec(
            name="ext",
            backend="external",
            config={},
            ready=ReadyProbe(kind="tcp", target=f"127.0.0.1:{port}"),
            endpoint=f"tcp://127.0.0.1:{port}",
        )
        await be.start(spec, ctx)
        assert await be.probe(spec) is HealthStatus.READY
    finally:
        sock.close()


@pytest.mark.anyio
async def test_external_probe_starting_when_target_down(ctx):
    be = ExternalBackend()
    spec = ServiceSpec(
        name="dn",
        backend="external",
        config={},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    await be.start(spec, ctx)
    assert await be.probe(spec) is HealthStatus.STARTING


@pytest.mark.anyio
async def test_external_probe_http_ready_when_endpoint_healthy(ctx):
    be = ExternalBackend()
    spec = ServiceSpec(
        name="ext-http",
        backend="external",
        config={},
        ready=ReadyProbe(kind="http", target="http://127.0.0.1:9999/health"),
        endpoint="http://127.0.0.1:9999",
    )

    with patch(
        "potpie_context_engine.adapters.outbound.managed_services.external_backend._http_probe",
        new=AsyncMock(return_value=True),
    ):
        assert await be.probe(spec) is HealthStatus.READY
