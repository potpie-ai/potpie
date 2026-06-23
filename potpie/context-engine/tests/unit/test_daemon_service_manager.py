import pathlib

import pytest

from context_engine.adapters.outbound.managed_services.subprocess_backend import SubprocessBackend
from context_engine.application.services.managed_service_manager import (
    DependencyCycle,
    ServiceManager,
    ServiceNotFound,
    ServiceStatus,
)
from context_engine.domain.ports.daemon.shell import (
    HealthStatus,
    ReadyProbe,
    ServiceBackend,
    ServiceSpec,
)
from context_engine.host.daemon_runtime.context import ShellContext
from context_engine.host.daemon_runtime.registry import Registry
from tests.conftest import free_port


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


def _backends() -> Registry[ServiceBackend]:
    r: Registry[ServiceBackend] = Registry()
    r.register("subprocess", lambda: SubprocessBackend())
    return r


def _status_for(mgr: ServiceManager, name: str) -> ServiceStatus:
    status = mgr.status(name)
    assert isinstance(status, ServiceStatus)
    return status


class _StopFailure(RuntimeError):
    pass


class _StopFailsBackend:
    async def start(self, spec: ServiceSpec, ctx: ShellContext) -> None:
        del spec, ctx
        return None

    async def stop(self, spec: ServiceSpec) -> None:
        raise _StopFailure("stop failed")

    async def probe(self, spec: ServiceSpec) -> HealthStatus:
        return HealthStatus.READY


def _failing_stop_backends(backend: _StopFailsBackend) -> Registry[ServiceBackend]:
    registry: Registry[ServiceBackend] = Registry()
    registry.register("failing", lambda: backend)
    return registry


@pytest.mark.anyio
async def test_up_waits_for_ready_and_registers_endpoint(
    tmp_path: pathlib.Path, daemon_ctx: ShellContext
):
    port = free_port()
    script = tmp_path / "srv.py"
    script.write_text(
        "import socket, sys\n"
        "s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen()\n"
        "while True:\n"
        "    c, _ = s.accept(); c.close()\n"
    )
    spec = _spec("a", ["python", str(script), str(port)], port)
    mgr = ServiceManager(specs={spec.name: spec}, backends=_backends(), ctx=daemon_ctx)
    try:
        await mgr.up("a")
        assert daemon_ctx.endpoints.get("a") == f"tcp://127.0.0.1:{port}"
        assert _status_for(mgr, "a").status is HealthStatus.READY
    finally:
        await mgr.down("a")


@pytest.mark.anyio
async def test_dependency_order(tmp_path: pathlib.Path, daemon_ctx: ShellContext):
    p1, p2 = free_port(), free_port()
    script = tmp_path / "srv.py"
    script.write_text(
        "import socket, sys\n"
        "s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen()\n"
        "while True:\n"
        "    c, _ = s.accept(); c.close()\n"
    )
    a = _spec("a", ["python", str(script), str(p1)], p1)
    b = _spec("b", ["python", str(script), str(p2)], p2, deps=["a"])
    mgr = ServiceManager(specs={"a": a, "b": b}, backends=_backends(), ctx=daemon_ctx)
    try:
        await mgr.up("b")  # transitively brings up "a" first
        assert _status_for(mgr, "a").status is HealthStatus.READY
        assert _status_for(mgr, "b").status is HealthStatus.READY
    finally:
        await mgr.down("b")
        await mgr.down("a")


@pytest.mark.anyio
async def test_cycle_detected(daemon_ctx: ShellContext):
    a = _spec("a", ["true"], 0, deps=["b"])
    b = _spec("b", ["true"], 0, deps=["a"])
    mgr = ServiceManager(specs={"a": a, "b": b}, backends=_backends(), ctx=daemon_ctx)
    with pytest.raises(DependencyCycle):
        await mgr.up("a")


@pytest.mark.anyio
async def test_unknown_service_raises(daemon_ctx: ShellContext):
    mgr = ServiceManager(specs={}, backends=_backends(), ctx=daemon_ctx)
    with pytest.raises(ServiceNotFound):
        await mgr.up("nope")


@pytest.mark.anyio
async def test_down_marks_degraded_and_keeps_endpoint_when_stop_fails(
    daemon_ctx: ShellContext,
):
    spec = ServiceSpec(
        name="bad",
        backend="failing",
        config={},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    mgr = ServiceManager(
        specs={spec.name: spec},
        backends=_failing_stop_backends(_StopFailsBackend()),
        ctx=daemon_ctx,
    )
    await mgr.up("bad")

    with pytest.raises(_StopFailure):
        await mgr.down("bad")

    assert _status_for(mgr, "bad").status is HealthStatus.DEGRADED
    assert mgr.started_names() == ["bad"]
    assert daemon_ctx.endpoints.get("bad") == spec.endpoint
