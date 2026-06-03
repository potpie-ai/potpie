import asyncio, logging, pathlib, pytest, httpx
from pydantic import BaseModel
from host.daemon_runtime.shell import DaemonRuntime, default_registries, BuiltinPluginsLoader
from host.daemon_runtime.config import (
    DaemonConfig, ShellSettings, TransportEntry, ComponentEntry,
)
from domain.ports.daemon.operations import OperationSpec, OperationContext, AuthRequirement
from domain.ports.daemon.shell import HealthStatus


class PingIn(BaseModel):
    nonce: int

class PingOut(BaseModel):
    nonce: int


async def ping(inp: PingIn, ctx: OperationContext) -> PingOut:
    return PingOut(nonce=inp.nonce)


class StubComponent:
    name = "stub"
    def __init__(self, **_cfg): pass
    async def on_start(self, ctx) -> None: return None
    async def on_stop(self) -> None: return None
    def health(self) -> HealthStatus: return HealthStatus.READY
    def operations(self) -> list:
        return [OperationSpec(
            name="stub.ping", input_model=PingIn, output_model=PingOut,
            handler=ping, summary="ping", auth=AuthRequirement.NONE,
        )]


@pytest.mark.asyncio
async def test_on_ready_fires_after_transport_serving(tmp_path: pathlib.Path):
    sock = tmp_path / "d.sock"
    cfg = DaemonConfig(
        shell=ShellSettings(data_dir=str(tmp_path), log_level="warning"),
        transports=[TransportEntry(type="http", bind=f"unix:{sock}")],
        services=[], components=[ComponentEntry(type="stub", requires_services=[], config={})],
    )
    regs = default_registries()
    regs.components.register("stub", lambda **cfg: StubComponent(**cfg))
    flags = {}
    def _on_ready():
        flags["ready"] = True
        flags["socket_existed"] = sock.exists()
    shell = DaemonRuntime(config=cfg, registries=regs, plugins_loader=BuiltinPluginsLoader(), on_ready=_on_ready)
    run_task = asyncio.create_task(shell.run())
    try:
        for _ in range(100):
            if flags.get("ready"):
                break
            await asyncio.sleep(0.02)
        assert flags.get("ready") is True
        assert flags.get("socket_existed") is True  # socket was bound before readiness was signalled
    finally:
        await shell.stop()
        await asyncio.wait_for(run_task, timeout=5.0)


@pytest.mark.asyncio
async def test_on_ready_not_called_when_startup_fails(tmp_path: pathlib.Path):
    sock = tmp_path / "d.sock"
    cfg = DaemonConfig(
        shell=ShellSettings(data_dir=str(tmp_path), log_level="warning"),
        transports=[TransportEntry(type="http", bind=f"unix:{sock}")],
        services=[], components=[ComponentEntry(type="boom", requires_services=[], config={})],
    )
    class BoomComponent:
        name = "boom"
        def __init__(self, **_): pass
        async def on_start(self, ctx): raise RuntimeError("startup boom")
        async def on_stop(self): return None
        def health(self): return HealthStatus.STOPPED
        def operations(self): return []
    regs = default_registries()
    regs.components.register("boom", lambda **cfg: BoomComponent(**cfg))
    called = {"ready": False}
    shell = DaemonRuntime(config=cfg, registries=regs, plugins_loader=BuiltinPluginsLoader(),
                  on_ready=lambda: called.__setitem__("ready", True))
    with pytest.raises(RuntimeError):
        await shell.run()
    assert called["ready"] is False


@pytest.mark.asyncio
async def test_shell_runs_and_serves_stub_component(tmp_path: pathlib.Path):
    sock = tmp_path / "d.sock"
    cfg = DaemonConfig(
        shell=ShellSettings(data_dir=str(tmp_path), log_level="warning"),
        transports=[TransportEntry(type="http", bind=f"unix:{sock}")],
        services=[],
        components=[ComponentEntry(type="stub", requires_services=[], config={})],
    )
    regs = default_registries()
    regs.components.register("stub", lambda **cfg: StubComponent(**cfg))

    shell = DaemonRuntime(config=cfg, registries=regs, plugins_loader=BuiltinPluginsLoader())
    run_task = asyncio.create_task(shell.run())
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.post("http://localhost/op/stub.ping", json={"nonce": 7})
            assert r.status_code == 200
            assert r.json() == {"nonce": 7}
    finally:
        await shell.stop()
        await asyncio.wait_for(run_task, timeout=5.0)
