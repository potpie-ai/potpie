from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import pytest

from potpie.runtime import (
    PROTOCOL_VERSION,
    ContextSelector,
    EngineOperation,
    EngineOperationRequest,
    HttpDaemonTransport,
    RuntimeEndpoint,
    TransportFailure,
    generate_bearer_token,
)
from potpie_context_engine.requests import RecordRequest


class _MalformedResponse:
    def json(self) -> dict[str, object]:
        return {"not": "a protocol response"}


class _MalformedResponseClient:
    async def post(self, *_args: object, **_kwargs: object) -> _MalformedResponse:
        return _MalformedResponse()


def test_tcp_endpoint_rejects_non_loopback_binding() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        RuntimeEndpoint(kind="tcp", address="0.0.0.0", port=7777)  # noqa: S104


def test_uds_endpoint_requires_absolute_path() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        RuntimeEndpoint(kind="uds", address="relative.sock")


def test_ipv6_loopback_endpoint_formats_valid_http_authority() -> None:
    endpoint = RuntimeEndpoint(kind="tcp", address="::1", port=7777)

    assert endpoint.display == "[::1]:7777"


def test_generated_bearer_tokens_are_unique_256_bit_secrets() -> None:
    first = generate_bearer_token()
    second = generate_bearer_token()

    assert first != second
    assert len(first.encode()) >= 32
    assert len(second.encode()) >= 32


@pytest.mark.anyio
async def test_schema_invalid_response_is_a_dispatched_transport_failure() -> None:
    transport = HttpDaemonTransport(
        endpoint=RuntimeEndpoint(kind="tcp", address="127.0.0.1", port=8765),
        bearer_token="x" * 32,
    )
    transport._client = _MalformedResponseClient()  # type: ignore[assignment]
    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="record-1",
        operation=EngineOperation.RECORD,
        selector=ContextSelector(kind="explicit", value="context-a"),
        payload=RecordRequest(),
        compatibility_ticket="ticket",
    )

    with pytest.raises(TransportFailure) as raised:
        await transport.send(request)

    assert raised.value.code == "response_envelope_malformed"
    assert raised.value.dispatched is True
