"""Additional http transport coverage: edge cases, admin endpoints, error paths."""
from __future__ import annotations
import asyncio, logging, pathlib, pytest
import httpx
from pydantic import BaseModel
from domain.ports.daemon.operations import (
    OperationRegistry, OperationSpec, OperationContext, OperationError, AuthRequirement,
)
from host.daemon_runtime.context import ShellContext, ServiceEndpoints
from host.daemon_runtime.health import HealthRegistrar
from host.daemon_runtime.ipc_auth import IpcAuthGate
from adapters.inbound.daemon_http.transport import HttpTransport
from domain.ports.daemon.shell import HealthStatus


class EchoIn(BaseModel):
    msg: str

class EchoOut(BaseModel):
    echoed: str


async def echo(inp: EchoIn, ctx: OperationContext) -> EchoOut:
    return EchoOut(echoed=inp.msg)


async def boom_generic(inp: EchoIn, ctx: OperationContext) -> EchoOut:
    raise RuntimeError("unexpected failure")


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={}, data_dir=tmp_path, logger=logging.getLogger("test"),
        endpoints=ServiceEndpoints(),
    )


@pytest.fixture
def ops() -> OperationRegistry:
    r = OperationRegistry()
    r.register(OperationSpec(
        name="echo.say", input_model=EchoIn, output_model=EchoOut,
        handler=echo, summary="echo", mutating=False, auth=AuthRequirement.NONE,
    ))
    r.register(OperationSpec(
        name="echo.boom", input_model=EchoIn, output_model=EchoOut,
        handler=boom_generic, summary="boom", mutating=False, auth=AuthRequirement.NONE,
    ))
    return r


def test_bind_raises_on_unsupported_scheme(ctx):
    t = HttpTransport(bind="ftp://something", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    with pytest.raises(ValueError, match="unsupported bind"):
        t.bind(ctx)


def test_health_returns_starting_before_serve(ctx):
    t = HttpTransport(bind="tcp:127.0.0.1:0", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    assert t.health() == HealthStatus.STARTING


@pytest.mark.asyncio
async def test_serve_raises_if_bind_not_called(ops):
    t = HttpTransport(bind="tcp:127.0.0.1:0", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    with pytest.raises(RuntimeError, match="bind"):
        await t.serve(ops)


@pytest.mark.asyncio
async def test_stop_before_serve_closes_sock(ctx):
    """Calling stop() before serve() should release the listening socket."""
    t = HttpTransport(bind="tcp:127.0.0.1:0", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    assert t._sock is not None
    await t.stop()
    assert t._sock is None


@pytest.mark.asyncio
async def test_unknown_operation_returns_404(tmp_path: pathlib.Path, ops, ctx):
    sock = tmp_path / "d.sock"
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.post("http://localhost/op/not.exist", json={})
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "not_found"
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_invalid_json_body_returns_400(tmp_path: pathlib.Path, ctx):
    """When the body fails pydantic validation, should get a 400."""
    sock = tmp_path / "d.sock"
    ops = OperationRegistry()
    ops.register(OperationSpec(
        name="echo.say", input_model=EchoIn, output_model=EchoOut,
        handler=echo, summary="echo", mutating=False, auth=AuthRequirement.NONE,
    ))
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            # Missing required "msg" field → ValidationError → 400
            r = await c.post("http://localhost/op/echo.say", json={})
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "invalid_input"
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_handler_exception_returns_500(tmp_path: pathlib.Path, ctx):
    sock = tmp_path / "d.sock"
    ops = OperationRegistry()
    ops.register(OperationSpec(
        name="echo.boom", input_model=EchoIn, output_model=EchoOut,
        handler=boom_generic, summary="boom", mutating=False, auth=AuthRequirement.NONE,
    ))
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.post("http://localhost/op/echo.boom", json={"msg": "x"})
            assert r.status_code == 500
            assert r.json()["error"]["code"] == "internal_error"
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_list_ops_endpoint(tmp_path: pathlib.Path, ops, ctx):
    sock = tmp_path / "d.sock"
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.get("http://localhost/op")
            assert r.status_code == 200
            assert "echo.say" in r.json()["operations"]
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_admin_health_endpoint(tmp_path: pathlib.Path, ops, ctx):
    sock = tmp_path / "d.sock"
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.get("http://localhost/admin/health")
            assert r.status_code == 200
            assert "status" in r.json()
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_admin_services_no_manager(tmp_path: pathlib.Path, ops, ctx):
    sock = tmp_path / "d.sock"
    # ctx.config has no "service_manager" key
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.get("http://localhost/admin/services")
            assert r.status_code == 200
            assert r.json()["services"] == []
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_admin_service_up_no_manager(tmp_path: pathlib.Path, ops, ctx):
    sock = tmp_path / "d.sock"
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.post("http://localhost/admin/services/myservice/up")
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "unavailable"
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_admin_service_down_no_manager(tmp_path: pathlib.Path, ops, ctx):
    sock = tmp_path / "d.sock"
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock))) as c:
            r = await c.post("http://localhost/admin/services/myservice/down")
            assert r.status_code == 503
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_tcp_no_auth_configured_anonymous_access(ctx):
    """TCP transport with no token configured allows anonymous access."""
    ops = OperationRegistry()
    ops.register(OperationSpec(
        name="echo.say", input_model=EchoIn, output_model=EchoOut,
        handler=echo, summary="echo", mutating=False, auth=AuthRequirement.NONE,
    ))
    t = HttpTransport(bind="tcp:127.0.0.1:0", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if t.bound_port():
                break
            await asyncio.sleep(0.05)
        port = t.bound_port()
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{port}/op/echo.say", json={"msg": "hi"})
            assert r.status_code == 200
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_uds_reuses_existing_sock_file(tmp_path: pathlib.Path, ops, ctx):
    """If a stale .sock file exists, bind should unlink it before creating new."""
    sock = tmp_path / "stale.sock"
    sock.write_text("stale")  # create a fake stale socket file
    t = HttpTransport(bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        assert sock.exists()  # real socket now exists
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
