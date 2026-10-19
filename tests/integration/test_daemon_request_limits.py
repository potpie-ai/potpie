"""The local daemon's request body limit is explicit and typed.

aiohttp's own default (1 MiB) used to refuse a large request, such as a
resource import of a big chunk directory, as a malformed envelope. The daemon
now takes ``max_request_bytes`` (64 MiB by default) and answers a larger body
with a typed 413 ``request_too_large``.
"""

# ruff: noqa: S101 - pytest integration tests use assertions intentionally.

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from potpie.runtime import (
    CanonicalDaemonRuntime,
    ContextSelector,
    DaemonEngineClient,
    HttpDaemonTransport,
    OperationCoordinator,
    RuntimeEndpoint,
    generate_bearer_token,
)
from potpie.runtime.endpoint import DEFAULT_MAX_REQUEST_BYTES
from potpie_context_engine import Success
from potpie_context_engine.core.agent_envelope import AgentEnvelope
from potpie_context_engine.requests import SearchRequest
from tests.conftest import free_port


class _Echo:
    async def handle(self, request, *, authentication):
        del authentication
        return Success(
            AgentEnvelope(
                pot_id="pot_1",
                intent="search",
                items=(),
                coverage=(),
                metadata={"query_length": len(request.payload.query)},
            )
        )


@asynccontextmanager
async def _runtime(tmp_path: Path, **options):
    endpoint = RuntimeEndpoint(kind="tcp", address="127.0.0.1", port=free_port())
    token = generate_bearer_token()
    runtime = CanonicalDaemonRuntime(
        endpoint=endpoint,
        bearer_token=token,
        operation_handler=_Echo(),
        ownership_lock_path=tmp_path / "runtime.lock",
        instance_id="instance-1",
        coordinator=OperationCoordinator(),
        **options,
    )
    await runtime.start()
    serve = asyncio.create_task(runtime.serve_until_shutdown())
    try:
        yield endpoint, token
    finally:
        await runtime.request_shutdown()
        await asyncio.wait_for(serve, timeout=2)


def test_the_default_limit_is_the_managed_host_limit() -> None:
    assert DEFAULT_MAX_REQUEST_BYTES == 64 * 1024 * 1024


@pytest.mark.anyio
async def test_a_request_over_one_mebibyte_is_served(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as (endpoint, token):
        transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
        client = DaemonEngineClient(
            selector=ContextSelector(kind="explicit", value="pot_1"),
            transport=transport,
            expected_instance_id="instance-1",
        )
        assert isinstance(await client.handshake(), Success)

        outcome = await client.search(SearchRequest(query="q" * (2 * 1024 * 1024)))

        assert isinstance(outcome, Success)
        assert outcome.value.metadata == {"query_length": 2 * 1024 * 1024}
        await transport.close()


@pytest.mark.anyio
async def test_a_request_over_the_limit_is_a_typed_413(tmp_path: Path) -> None:
    async with _runtime(tmp_path, max_request_bytes=4096) as (endpoint, token):
        async with httpx.AsyncClient(
            base_url=f"http://{endpoint.address}:{endpoint.port}"
        ) as client:
            response = await client.post(
                "/v1/operations",
                json={"request_id": "big-1", "padding": "x" * 8192},
                headers={"Authorization": f"Bearer {token}"},
            )

    body = response.json()
    assert response.status_code == 413
    assert body["outcome"]["error"]["category"] == "protocol"
    assert body["outcome"]["error"]["code"] == "request_too_large"
    assert body["outcome"]["error"]["details"] == {"max_bytes": 4096}
