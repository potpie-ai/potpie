import asyncio
import logging
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from pydantic import BaseModel

from adapters.inbound.daemon_http.transport import HttpTransport
from domain.ports.daemon.operations import (
    AuthRequirement,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationSpec,
)
from host.daemon_runtime.context import ServiceEndpoints, ShellContext
from host.daemon_runtime.health import HealthRegistrar
from host.daemon_runtime.ipc_auth import IpcAuthGate
from tests.conftest import wait_for_condition


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
    r.register(
        OperationSpec(
            name="echo.say",
            input_model=EchoIn,
            output_model=EchoOut,
            handler=echo,
            summary="echo",
            mutating=False,
            auth=AuthRequirement.NONE,
        )
    )
    r.register(
        OperationSpec(
            name="echo.boom",
            input_model=EchoIn,
            output_model=EchoOut,
            handler=boom,
            summary="boom",
            mutating=False,
            auth=AuthRequirement.NONE,
        )
    )
    return r


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("test"),
        endpoints=ServiceEndpoints(),
    )


@asynccontextmanager
async def run_transport(
    transport: HttpTransport, ops: OperationRegistry
) -> AsyncIterator[None]:
    task = asyncio.create_task(transport.serve(ops))
    try:
        yield
    finally:
        await transport.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.anyio
async def test_uds_dispatch_success(short_socket_dir: pathlib.Path, ops, ctx):
    sock = short_socket_dir / "d.sock"
    t = HttpTransport(
        bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar()
    )
    t.bind(ctx)
    async with run_transport(t, ops):
        await wait_for_condition(
            lambda: sock.exists(),
            error_message=f"socket {sock} did not appear",
        )
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(sock))
        ) as c:
            r = await c.post("http://localhost/op/echo.say", json={"msg": "hi"})
            assert r.status_code == 200
            assert r.json() == {"echoed": "hi"}


@pytest.mark.anyio
async def test_error_maps_to_http_status(short_socket_dir: pathlib.Path, ops, ctx):
    sock = short_socket_dir / "d.sock"
    t = HttpTransport(
        bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar()
    )
    t.bind(ctx)
    async with run_transport(t, ops):
        await wait_for_condition(
            lambda: sock.exists(),
            error_message=f"socket {sock} did not appear",
        )
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(sock))
        ) as c:
            r = await c.post("http://localhost/op/echo.boom", json={"msg": "x"})
            assert r.status_code == 404
            body = r.json()
            assert body["error"]["code"] == "not_found"
            assert body["error"]["message"] == "missing"
            assert body["error"]["detail"] == {"k": "x"}


@pytest.mark.anyio
async def test_tcp_bind_works(ops, ctx):
    t = HttpTransport(
        bind="tcp:127.0.0.1:0", auth=IpcAuthGate(token=None), health=HealthRegistrar()
    )
    t.bind(ctx)
    async with run_transport(t, ops):
        await wait_for_condition(
            lambda: t.bound_port() is not None,
            error_message="tcp transport did not bind to a port",
        )
        port = t.bound_port()
        assert port is not None and port > 0
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{port}/op/echo.say", json={"msg": "ok"})
            assert r.status_code == 200


@pytest.mark.anyio
async def test_tcp_required_auth_rejected_without_token(ctx):
    reg = OperationRegistry()

    async def secret(inp: EchoIn, c: OperationContext) -> EchoOut:
        return EchoOut(echoed=inp.msg)

    reg.register(
        OperationSpec(
            name="echo.secret",
            input_model=EchoIn,
            output_model=EchoOut,
            handler=secret,
            summary="s",
            auth=AuthRequirement.REQUIRED,
        )
    )
    t = HttpTransport(
        bind="tcp:127.0.0.1:0",
        auth=IpcAuthGate(token="tok-abc"),
        health=HealthRegistrar(),
    )
    t.bind(ctx)
    async with run_transport(t, reg):
        await wait_for_condition(
            lambda: t.bound_port() is not None,
            error_message="tcp auth transport did not bind to a port",
        )
        port = t.bound_port()
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"http://127.0.0.1:{port}/op/echo.secret", json={"msg": "x"}
            )
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "unauthorized"
            r = await c.post(
                f"http://127.0.0.1:{port}/op/echo.secret",
                json={"msg": "x"},
                headers={"x-potpie-token": "tok-abc"},
            )
            assert r.status_code == 200
            assert r.json() == {"echoed": "x"}
