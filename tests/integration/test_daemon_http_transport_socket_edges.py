from __future__ import annotations

import asyncio
import logging
import pathlib

import httpx
import pytest
from pydantic import BaseModel

from potpie.daemon.http.transport import HttpTransport
from potpie_context_engine.domain.ports.daemon.operations import (
    AuthRequirement,
    OperationContext,
    OperationRegistry,
    OperationSpec,
)
from potpie.daemon.runtime.context import ServiceEndpoints, ShellContext
from potpie.daemon.runtime.health import HealthRegistrar
from potpie.daemon.runtime.ipc_auth import IpcAuthGate


class EchoIn(BaseModel):
    msg: str


class EchoOut(BaseModel):
    echoed: str


async def echo(inp: EchoIn, ctx: OperationContext) -> EchoOut:
    return EchoOut(echoed=inp.msg)


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("test"),
        endpoints=ServiceEndpoints(),
    )


@pytest.fixture
def ops() -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(
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
    return registry


@pytest.mark.anyio
async def test_tcp_no_auth_configured_anonymous_access(ctx):
    ops = OperationRegistry()
    ops.register(
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
    transport = HttpTransport(
        bind="tcp:127.0.0.1:0",
        auth=IpcAuthGate(token=None),
        health=HealthRegistrar(),
    )
    transport.bind(ctx)
    task = asyncio.create_task(transport.serve(ops))
    try:
        for _ in range(50):
            if transport.bound_port():
                break
            await asyncio.sleep(0.05)
        port = transport.bound_port()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/op/echo.say", json={"msg": "hi"}
            )
            assert response.status_code == 200
    finally:
        await transport.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.anyio
async def test_uds_reuses_existing_sock_file(short_socket_dir: pathlib.Path, ops, ctx):
    sock = short_socket_dir / "stale.sock"
    sock.write_text("stale")
    transport = HttpTransport(
        bind=f"unix:{sock}", auth=IpcAuthGate(token=None), health=HealthRegistrar()
    )
    transport.bind(ctx)
    task = asyncio.create_task(transport.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        assert sock.exists()
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(sock))
        ) as client:
            response = await client.post(
                "http://localhost/op/echo.say", json={"msg": "hi"}
            )
            assert response.status_code == 200
    finally:
        await transport.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
