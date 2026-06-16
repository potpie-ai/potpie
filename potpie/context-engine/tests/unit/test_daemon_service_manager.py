import logging
import pathlib
import socket
import pytest
from application.services.managed_service_manager import (
    ServiceManager,
    ServiceNotFound,
    DependencyCycle,
)
from adapters.outbound.managed_services.subprocess_backend import SubprocessBackend
from domain.ports.daemon.shell import ServiceSpec, ReadyProbe, HealthStatus
from host.daemon_runtime.context import ShellContext, ServiceEndpoints
from host.daemon_runtime.registry import Registry


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _spec(
    name: str, cmd: list[str], port: int, deps: list[str] | None = None
) -> ServiceSpec:
    return ServiceSpec(
        name=name,
        backend="subprocess",
        config={"command": cmd},
        ready=ReadyProbe(
            kind="tcp", target=f"127.0.0.1:{port}", interval_s=0.2, timeout_s=10.0
        ),
        endpoint=f"tcp://127.0.0.1:{port}",
        depends_on=deps or [],
    )


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("t"),
        endpoints=ServiceEndpoints(),
    )


def _backends() -> Registry:
    r: Registry = Registry()
    r.register("subprocess", lambda: SubprocessBackend())
    return r


@pytest.mark.anyio
async def test_up_waits_for_ready_and_registers_endpoint(tmp_path: pathlib.Path, ctx):
    port = _free_port()
    script = tmp_path / "srv.py"
    script.write_text(
        "import socket, sys\n"
        "s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen()\n"
        "while True:\n"
        "    c, _ = s.accept(); c.close()\n"
    )
    spec = _spec("a", ["python", str(script), str(port)], port)
    mgr = ServiceManager(specs={spec.name: spec}, backends=_backends(), ctx=ctx)
    try:
        await mgr.up("a")
        assert ctx.endpoints.get("a") == f"tcp://127.0.0.1:{port}"
        assert mgr.status("a").status is HealthStatus.READY
    finally:
        await mgr.down("a")


@pytest.mark.anyio
async def test_dependency_order(tmp_path: pathlib.Path, ctx):
    p1, p2 = _free_port(), _free_port()
    script = tmp_path / "srv.py"
    script.write_text(
        "import socket, sys\n"
        "s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen()\n"
        "while True:\n"
        "    c, _ = s.accept(); c.close()\n"
    )
    a = _spec("a", ["python", str(script), str(p1)], p1)
    b = _spec("b", ["python", str(script), str(p2)], p2, deps=["a"])
    mgr = ServiceManager(specs={"a": a, "b": b}, backends=_backends(), ctx=ctx)
    try:
        await mgr.up("b")  # transitively brings up "a" first
        assert mgr.status("a").status is HealthStatus.READY
        assert mgr.status("b").status is HealthStatus.READY
    finally:
        await mgr.down("b")
        await mgr.down("a")


@pytest.mark.anyio
async def test_cycle_detected(ctx):
    a = _spec("a", ["true"], 0, deps=["b"])
    b = _spec("b", ["true"], 0, deps=["a"])
    mgr = ServiceManager(specs={"a": a, "b": b}, backends=_backends(), ctx=ctx)
    with pytest.raises(DependencyCycle):
        await mgr.up("a")


@pytest.mark.anyio
async def test_unknown_service_raises(ctx):
    mgr = ServiceManager(specs={}, backends=_backends(), ctx=ctx)
    with pytest.raises(ServiceNotFound):
        await mgr.up("nope")
