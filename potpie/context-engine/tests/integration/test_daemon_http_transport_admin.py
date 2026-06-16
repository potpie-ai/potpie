from __future__ import annotations

import asyncio
import logging
import pathlib

import httpx
import pytest
from pydantic import BaseModel

from adapters.inbound.daemon_http.transport import HttpTransport
from domain.ports.daemon.operations import (
    AuthRequirement,
    OperationContext,
    OperationRegistry,
    OperationSpec,
)
from host.daemon_runtime.context import ServiceEndpoints, ShellContext
from host.daemon_runtime.health import HealthRegistrar
from host.daemon_runtime.ipc_auth import IpcAuthGate


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
async def test_admin_health_endpoint(short_socket_dir: pathlib.Path, ops, ctx):
    sock = short_socket_dir / "d.sock"
    transport = HttpTransport(
        bind=f"unix:{sock}",
        auth=IpcAuthGate(token=None),
        health=HealthRegistrar(),
    )
    transport.bind(ctx)
    task = asyncio.create_task(transport.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(sock))
        ) as client:
            response = await client.get("http://localhost/admin/health")
            assert response.status_code == 200
            assert "status" in response.json()
    finally:
        await transport.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.anyio
async def test_admin_services_no_manager(short_socket_dir: pathlib.Path, ops, ctx):
    sock = short_socket_dir / "d.sock"
    transport = HttpTransport(
        bind=f"unix:{sock}",
        auth=IpcAuthGate(token=None),
        health=HealthRegistrar(),
    )
    transport.bind(ctx)
    task = asyncio.create_task(transport.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(sock))
        ) as client:
            response = await client.get("http://localhost/admin/services")
            assert response.status_code == 200
            assert response.json()["services"] == []
    finally:
        await transport.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.anyio
async def test_admin_service_up_no_manager(short_socket_dir: pathlib.Path, ops, ctx):
    sock = short_socket_dir / "d.sock"
    transport = HttpTransport(
        bind=f"unix:{sock}",
        auth=IpcAuthGate(token=None),
        health=HealthRegistrar(),
    )
    transport.bind(ctx)
    task = asyncio.create_task(transport.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(sock))
        ) as client:
            response = await client.post("http://localhost/admin/services/myservice/up")
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "unavailable"
    finally:
        await transport.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.anyio
async def test_admin_service_down_no_manager(short_socket_dir: pathlib.Path, ops, ctx):
    sock = short_socket_dir / "d.sock"
    transport = HttpTransport(
        bind=f"unix:{sock}",
        auth=IpcAuthGate(token=None),
        health=HealthRegistrar(),
    )
    transport.bind(ctx)
    task = asyncio.create_task(transport.serve(ops))
    try:
        for _ in range(50):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(sock))
        ) as client:
            response = await client.post(
                "http://localhost/admin/services/myservice/down"
            )
            assert response.status_code == 503
    finally:
        await transport.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
