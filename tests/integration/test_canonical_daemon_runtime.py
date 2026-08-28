from __future__ import annotations

# ruff: noqa: S101 - pytest integration tests use assertions intentionally.

import asyncio
import stat
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from potpie.runtime import (
    PROTOCOL_VERSION,
    AuthenticationError,
    CanonicalDaemonRuntime,
    ContextSelector,
    DaemonControlClient,
    DaemonEngineClient,
    DaemonInternalError,
    EngineOperation,
    EngineOperationRequest,
    FailureResponse,
    HttpDaemonTransport,
    OperationCoordinator,
    ProtocolError,
    ResourceLifecycleError,
    RuntimeEndpoint,
    RuntimeOwnershipLock,
    generate_bearer_token,
)
from potpie_context_engine import Failure, Success
from potpie_context_engine.core.agent_envelope import AgentEnvelope
from potpie_context_engine.requests import SearchRequest
from tests.conftest import free_port


class _Handler:
    def __init__(self) -> None:
        self.calls: list[tuple[EngineOperation, object]] = []
        self.raise_defect = False
        self.unsupported_result = False
        self.entered = asyncio.Event()
        self.gate: asyncio.Event | None = None

    async def handle(self, request: EngineOperationRequest, *, authentication: object):
        self.calls.append((request.operation, authentication))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.raise_defect:
            raise RuntimeError("sensitive traceback detail")
        if self.unsupported_result:
            return Success(object())
        return Success(
            AgentEnvelope(
                pot_id=request.selector.value or "",
                intent="search",
                items=(),
                coverage=(),
                metadata={"query": request.payload.query},
            )
        )


def _endpoint(kind: str, tmp_path: Path) -> RuntimeEndpoint:
    if kind == "uds":
        return RuntimeEndpoint(kind="uds", address=str(tmp_path / "runtime.sock"))
    return RuntimeEndpoint(kind="tcp", address="127.0.0.1", port=free_port())


@asynccontextmanager
async def _running_runtime(kind: str, tmp_path: Path, handler: _Handler):
    endpoint = _endpoint(kind, tmp_path)
    token = generate_bearer_token()
    runtime = CanonicalDaemonRuntime(
        endpoint=endpoint,
        bearer_token=token,
        operation_handler=handler,
        ownership_lock_path=tmp_path / "runtime.lock",
        instance_id="instance-1",
        backend_profile="embedded",
        ui_url="http://127.0.0.1:8765",
        coordinator=OperationCoordinator(),
    )
    await runtime.start()
    serve_task = asyncio.create_task(runtime.serve_until_shutdown())
    try:
        yield runtime, serve_task, endpoint, token
    finally:
        if handler.gate is not None:
            handler.gate.set()
        await runtime.request_shutdown()
        await asyncio.wait_for(serve_task, timeout=2)


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["tcp", "uds"])
async def test_authenticated_runtime_handshake_operation_and_typed_stop(
    kind: str, tmp_path: Path, short_socket_dir: Path
) -> None:
    handler = _Handler()
    runtime_dir = short_socket_dir if kind == "uds" else tmp_path
    async with _running_runtime(kind, runtime_dir, handler) as (
        runtime,
        serve_task,
        endpoint,
        token,
    ):
        transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
        engine_client = DaemonEngineClient(
            selector=ContextSelector(kind="explicit", value="context-a"),
            transport=transport,
            expected_instance_id="instance-1",
        )
        control_client = DaemonControlClient(
            transport=transport,
            expected_instance_id="instance-1",
        )

        handshake = await engine_client.handshake()
        operation = await engine_client.search(SearchRequest(query="typed"))
        control_handshake = await control_client.handshake()
        status = await control_client.status()
        shutdown = await control_client.shutdown(reason="integration_test")

        assert isinstance(handshake, Success)
        assert isinstance(control_handshake, Success)
        assert handshake.value.compatibility_ticket
        assert control_handshake.value.compatibility_ticket
        assert (
            handshake.value.compatibility_ticket
            != control_handshake.value.compatibility_ticket
        )
        assert isinstance(status, Success)
        assert status.value.backend_profile == "embedded"
        assert status.value.ui_url == "http://127.0.0.1:8765"
        assert isinstance(operation, Success)
        assert operation.value == AgentEnvelope(
            pot_id="context-a",
            intent="search",
            items=(),
            coverage=(),
            metadata={"query": "typed"},
        )
        assert isinstance(shutdown, Success)
        assert shutdown.value.accepted is True
        await asyncio.wait_for(serve_task, timeout=2)
        assert runtime.lifecycle_state == "stopped"
        assert handler.calls[0][0] is EngineOperation.SEARCH
        await transport.close()

        if kind == "uds":
            assert not Path(endpoint.address).exists()


@pytest.mark.anyio
async def test_uds_socket_is_owner_only_while_runtime_is_ready(
    short_socket_dir: Path,
) -> None:
    handler = _Handler()
    async with _running_runtime("uds", short_socket_dir, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        _token,
    ):
        mode = stat.S_IMODE(Path(endpoint.address).stat().st_mode)
        parent_mode = stat.S_IMODE(Path(endpoint.address).parent.stat().st_mode)

        assert mode == 0o600
        assert parent_mode == 0o700


@pytest.mark.anyio
async def test_bad_bearer_token_returns_typed_authentication_failure(
    tmp_path: Path,
) -> None:
    handler = _Handler()
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        _token,
    ):
        transport = HttpDaemonTransport(
            endpoint=endpoint,
            bearer_token="x" * 32,
        )
        client = DaemonControlClient(
            transport=transport,
            expected_instance_id="instance-1",
        )

        outcome = await client.handshake()

        assert isinstance(outcome, Failure)
        assert isinstance(outcome.error, AuthenticationError)
        assert outcome.error.code == "daemon_authentication_failed"
        await transport.close()


@pytest.mark.anyio
async def test_catalog_mismatch_is_rejected_during_handshake(tmp_path: Path) -> None:
    handler = _Handler()
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        token,
    ):
        base_url = f"http://{endpoint.address}:{endpoint.port}"
        async with httpx.AsyncClient(base_url=base_url) as client:
            response = await client.post(
                "/v1/operations",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": "catalog-mismatch-1",
                    "operation": "daemon.handshake",
                    "payload": {
                        "client_protocol_min": PROTOCOL_VERSION,
                        "client_protocol_max": PROTOCOL_VERSION,
                        "expected_instance_id": "instance-1",
                        "client_operation_catalog_fingerprint": "different",
                    },
                },
            )

        assert response.status_code == 409
        assert response.json()["outcome"]["error"]["code"] == (
            "operation_catalog_mismatch"
        )
        assert handler.calls == []


@pytest.mark.anyio
async def test_incompatible_protocol_version_returns_correlated_typed_failure(
    tmp_path: Path,
) -> None:
    handler = _Handler()
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        token,
    ):
        base_url = f"http://{endpoint.address}:{endpoint.port}"
        headers = {"Authorization": f"Bearer {token}"}
        request_id = "incompatible-version-1"
        async with httpx.AsyncClient(base_url=base_url) as client:
            response = await client.post(
                "/v1/operations",
                json={
                    "protocol_version": PROTOCOL_VERSION + 1,
                    "request_id": request_id,
                    "operation": "search",
                    "selector": {"kind": "explicit", "value": "context-a"},
                    "payload": {"query": "typed"},
                    "compatibility_ticket": None,
                },
                headers=headers,
            )

        body = response.json()
        assert response.status_code == 409
        assert body["protocol_version"] == PROTOCOL_VERSION + 1
        assert body["request_id"] == request_id
        assert body["outcome"]["error"]["category"] == "protocol"
        assert body["outcome"]["error"]["code"] == "protocol_version_incompatible"
        assert "traceback" not in str(body).lower()
        assert handler.calls == []


@pytest.mark.anyio
async def test_domain_operation_is_rejected_before_handshake(tmp_path: Path) -> None:
    handler = _Handler()
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        token,
    ):
        transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
        request = EngineOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id="search-1",
            operation=EngineOperation.SEARCH,
            selector=ContextSelector(kind="explicit", value="context-a"),
            payload=SearchRequest(),
        )

        response = await transport.send(request)

        assert isinstance(response, FailureResponse)
        assert isinstance(response.outcome.error, ProtocolError)
        assert response.outcome.error.code == "compatibility_ticket_invalid"
        assert handler.calls == []
        await transport.close()


@pytest.mark.anyio
async def test_invalid_compatibility_ticket_is_rejected(tmp_path: Path) -> None:
    handler = _Handler()
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        token,
    ):
        transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
        request = EngineOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id="search-invalid-ticket",
            operation=EngineOperation.SEARCH,
            selector=ContextSelector(kind="explicit", value="context-a"),
            payload=SearchRequest(),
            compatibility_ticket="not-a-valid-ticket",
        )

        response = await transport.send(request)

        assert isinstance(response, FailureResponse)
        assert response.outcome.error.code == "compatibility_ticket_invalid"
        assert handler.calls == []
        await transport.close()


@pytest.mark.anyio
async def test_malformed_and_unknown_requests_return_safe_correlated_envelopes(
    tmp_path: Path,
) -> None:
    handler = _Handler()
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        token,
    ):
        base_url = f"http://{endpoint.address}:{endpoint.port}"
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(base_url=base_url) as client:
            malformed = await client.post(
                "/v1/operations", json={"not": "an envelope"}, headers=headers
            )
            unknown = await client.post(
                "/v1/operations",
                json={
                    "protocol_version": 1,
                    "request_id": "unknown-1",
                    "operation": "runtime.reflect",
                    "payload": {},
                },
                headers=headers,
            )

        malformed_body = malformed.json()
        unknown_body = unknown.json()
        assert malformed.status_code == 400
        assert isinstance(malformed_body["request_id"], str)
        assert malformed_body["request_id"]
        assert (
            malformed_body["outcome"]["error"]["code"] == "request_envelope_malformed"
        )
        assert unknown.status_code == 400
        assert unknown_body["request_id"] == "unknown-1"
        assert unknown_body["outcome"]["error"]["code"] == "operation_unknown"
        assert "traceback" not in str(malformed_body).lower()
        assert "traceback" not in str(unknown_body).lower()


@pytest.mark.anyio
async def test_internal_handler_failure_is_redacted(tmp_path: Path) -> None:
    handler = _Handler()
    handler.raise_defect = True
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        token,
    ):
        transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
        client = DaemonEngineClient(
            selector=ContextSelector(kind="explicit", value="context-a"),
            transport=transport,
            expected_instance_id="instance-1",
        )
        assert isinstance(await client.handshake(), Success)

        outcome = await client.search(SearchRequest())

        assert isinstance(outcome, Failure)
        assert isinstance(outcome.error, DaemonInternalError)
        assert outcome.error.code == "daemon_internal_failure"
        assert "sensitive" not in outcome.error.message
        assert "traceback" not in str(outcome.error.details).lower()
        await transport.close()


@pytest.mark.anyio
async def test_unencodable_handler_result_returns_typed_internal_failure(
    tmp_path: Path,
) -> None:
    handler = _Handler()
    handler.unsupported_result = True
    async with _running_runtime("tcp", tmp_path, handler) as (
        _runtime,
        _serve_task,
        endpoint,
        token,
    ):
        transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
        client = DaemonEngineClient(
            selector=ContextSelector(kind="explicit", value="context-a"),
            transport=transport,
            expected_instance_id="instance-1",
        )
        assert isinstance(await client.handshake(), Success)

        outcome = await client.search(SearchRequest())

        assert isinstance(outcome, Failure)
        assert isinstance(outcome.error, DaemonInternalError)
        assert outcome.error.code == "daemon_response_encoding_failed"
        await transport.close()


@pytest.mark.anyio
async def test_typed_shutdown_drains_active_operation(tmp_path: Path) -> None:
    handler = _Handler()
    handler.gate = asyncio.Event()
    async with _running_runtime("tcp", tmp_path, handler) as (
        runtime,
        serve_task,
        endpoint,
        token,
    ):
        transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
        engine_client = DaemonEngineClient(
            selector=ContextSelector(kind="explicit", value="context-a"),
            transport=transport,
            expected_instance_id="instance-1",
        )
        control_client = DaemonControlClient(
            transport=transport,
            expected_instance_id="instance-1",
        )
        assert isinstance(await engine_client.handshake(), Success)
        assert isinstance(await control_client.handshake(), Success)
        operation_task = asyncio.create_task(engine_client.search(SearchRequest()))
        await handler.entered.wait()

        shutdown = await control_client.shutdown(reason="drain_test")
        await asyncio.sleep(0)

        assert isinstance(shutdown, Success)
        assert runtime.lifecycle_state == "draining"
        assert not serve_task.done()

        handler.gate.set()
        assert isinstance(await operation_task, Success)
        await asyncio.wait_for(serve_task, timeout=2)
        assert runtime.lifecycle_state == "stopped"
        await transport.close()


@pytest.mark.anyio
async def test_shutdown_cleanup_failure_releases_runtime_ownership(
    tmp_path: Path,
) -> None:
    handler = _Handler()
    endpoint = _endpoint("tcp", tmp_path)
    token = generate_bearer_token()
    lock_path = tmp_path / "runtime.lock"
    ownership_during_cleanup: list[object] = []

    async def failed_cleanup():
        return Failure(
            ResourceLifecycleError(
                code="cleanup_failed",
                message="cleanup failed",
            )
        )

    def cleanup_runtime_records(_ownership: RuntimeOwnershipLock) -> None:
        contender = RuntimeOwnershipLock(lock_path)
        ownership_during_cleanup.append(contender.acquire())
        contender.release()

    runtime = CanonicalDaemonRuntime(
        endpoint=endpoint,
        bearer_token=token,
        operation_handler=handler,
        ownership_lock_path=lock_path,
        instance_id="instance-1",
        shutdown_resources=failed_cleanup,
        before_ownership_release=cleanup_runtime_records,
        coordinator=OperationCoordinator(),
    )
    await runtime.start()
    serve_task = asyncio.create_task(runtime.serve_until_shutdown())
    transport = HttpDaemonTransport(endpoint=endpoint, bearer_token=token)
    control = DaemonControlClient(
        transport=transport,
        expected_instance_id="instance-1",
    )
    assert isinstance(await control.handshake(), Success)
    assert isinstance(await control.shutdown(reason="cleanup_test"), Success)

    with pytest.raises(RuntimeError, match="resource shutdown failed"):
        await serve_task

    assert runtime.lifecycle_state == "stopped"
    assert len(ownership_during_cleanup) == 1
    assert isinstance(ownership_during_cleanup[0], Failure)
    ownership = RuntimeOwnershipLock(lock_path)
    assert isinstance(ownership.acquire(), Success)
    ownership.release()
    await transport.close()
