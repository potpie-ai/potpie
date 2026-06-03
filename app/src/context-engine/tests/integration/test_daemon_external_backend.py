import asyncio, logging, pathlib, socket, pytest
from adapters.outbound.managed_services.external_backend import ExternalBackend
from domain.ports.daemon.shell import ServiceSpec, ReadyProbe, HealthStatus
from host.daemon_runtime.context import ShellContext, ServiceEndpoints


def _free_port_and_listener() -> tuple[int, socket.socket]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s.getsockname()[1], s


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={}, data_dir=tmp_path, logger=logging.getLogger("t"),
        endpoints=ServiceEndpoints(),
    )


@pytest.mark.asyncio
async def test_external_does_not_start_anything(ctx):
    be = ExternalBackend()
    spec = ServiceSpec(
        name="x", backend="external", config={},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    await be.start(spec, ctx)
    await be.stop(spec)


@pytest.mark.asyncio
async def test_external_probe_ready_when_target_listening(ctx):
    port, sock = _free_port_and_listener()
    try:
        be = ExternalBackend()
        spec = ServiceSpec(
            name="ext", backend="external", config={},
            ready=ReadyProbe(kind="tcp", target=f"127.0.0.1:{port}"),
            endpoint=f"tcp://127.0.0.1:{port}",
        )
        await be.start(spec, ctx)
        assert await be.probe(spec) is HealthStatus.READY
    finally:
        sock.close()


@pytest.mark.asyncio
async def test_external_probe_starting_when_target_down(ctx):
    be = ExternalBackend()
    spec = ServiceSpec(
        name="dn", backend="external", config={},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    await be.start(spec, ctx)
    assert await be.probe(spec) is HealthStatus.STARTING
