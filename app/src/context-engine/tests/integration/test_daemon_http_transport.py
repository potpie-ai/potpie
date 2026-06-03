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


class EchoIn(BaseModel):
    msg: str

class EchoOut(BaseModel):
    echoed: str


async def echo(inp: EchoIn, ctx: OperationContext) -> EchoOut:
    return EchoOut(echoed=inp.msg)


async def boom(inp: EchoIn, ctx: OperationContext) -> EchoOut:
    raise OperationError("not_found", "missing", detail={"k": inp.msg})


@pytest.fixture
def ops() -> OperationRegistry:
    r = OperationRegistry()
    r.register(OperationSpec(
        name="echo.say", input_model=EchoIn, output_model=EchoOut,
        handler=echo, summary="echo", mutating=False, auth=AuthRequirement.NONE,
    ))
    r.register(OperationSpec(
        name="echo.boom", input_model=EchoIn, output_model=EchoOut,
        handler=boom, summary="boom", mutating=False, auth=AuthRequirement.NONE,
    ))
    return r


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={}, data_dir=tmp_path, logger=logging.getLogger("test"),
        endpoints=ServiceEndpoints(),
    )


@pytest.mark.asyncio
async def test_uds_dispatch_success(tmp_path: pathlib.Path, ops, ctx):
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
            r = await c.post("http://localhost/op/echo.say", json={"msg": "hi"})
            assert r.status_code == 200
            assert r.json() == {"echoed": "hi"}
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_error_maps_to_http_status(tmp_path: pathlib.Path, ops, ctx):
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
            r = await c.post("http://localhost/op/echo.boom", json={"msg": "x"})
            assert r.status_code == 404
            body = r.json()
            assert body["error"]["code"] == "not_found"
            assert body["error"]["message"] == "missing"
            assert body["error"]["detail"] == {"k": "x"}
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_tcp_bind_works(ops, ctx):
    t = HttpTransport(bind="tcp:127.0.0.1:0", auth=IpcAuthGate(token=None), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(ops))
    try:
        for _ in range(50):
            if t.bound_port():
                break
            await asyncio.sleep(0.05)
        port = t.bound_port()
        assert port is not None and port > 0
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{port}/op/echo.say", json={"msg": "ok"})
            assert r.status_code == 200
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_tcp_required_auth_rejected_without_token(ctx):
    reg = OperationRegistry()

    async def secret(inp: EchoIn, c: OperationContext) -> EchoOut:
        return EchoOut(echoed=inp.msg)

    reg.register(OperationSpec(
        name="echo.secret", input_model=EchoIn, output_model=EchoOut,
        handler=secret, summary="s", auth=AuthRequirement.REQUIRED,
    ))
    t = HttpTransport(bind="tcp:127.0.0.1:0", auth=IpcAuthGate(token="tok-abc"), health=HealthRegistrar())
    t.bind(ctx)
    task = asyncio.create_task(t.serve(reg))
    try:
        for _ in range(50):
            if t.bound_port():
                break
            await asyncio.sleep(0.05)
        port = t.bound_port()
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{port}/op/echo.secret", json={"msg": "x"})
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "unauthorized"
            r = await c.post(
                f"http://127.0.0.1:{port}/op/echo.secret", json={"msg": "x"},
                headers={"x-potpie-token": "tok-abc"},
            )
            assert r.status_code == 200
            assert r.json() == {"echoed": "x"}
    finally:
        await t.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
