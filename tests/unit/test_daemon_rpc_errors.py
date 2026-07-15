from __future__ import annotations

from typing import cast

import httpx
import pytest

from potpie.daemon import client as daemon_client
from potpie.daemon.client import DaemonRpcTransport
from potpie.daemon.lifecycle import Daemon
from potpie.daemon.rpc_errors import decode_domain_error, encode_domain_error
from potpie.runtime.contracts import RegisterRepoSourceRequest
from potpie.runtime.errors import DaemonRpcFailure
from potpie_context_engine.contracts import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)


@pytest.mark.parametrize(
    ("exception", "code", "exception_type"),
    (
        (
            CapabilityNotImplemented(
                "engine.ledger.query",
                detail="managed ledger is unavailable",
                recommended_next_action="configure a ledger binding",
            ),
            "ENGINE_CAPABILITY_NOT_IMPLEMENTED",
            CapabilityNotImplemented,
        ),
        (
            PotNotFound("No pot matching 'missing'."),
            "ENGINE_POT_NOT_FOUND",
            PotNotFound,
        ),
        (
            ContextEngineDisabled("Context engine disabled."),
            "ENGINE_DISABLED",
            ContextEngineDisabled,
        ),
        (ValueError("invalid graph view"), "ENGINE_VALIDATION_ERROR", ValueError),
    ),
)
def test_domain_errors_round_trip_with_local_exception_family(
    exception: Exception, code: str, exception_type: type[Exception]
) -> None:
    encoded = encode_domain_error(exception)

    assert encoded is not None
    assert encoded.code == code
    decoded = decode_domain_error(
        code=encoded.code, message=encoded.message, details=encoded.details
    )
    assert type(decoded) is exception_type
    assert str(decoded) == str(exception)


def test_capability_error_preserves_guidance() -> None:
    original = CapabilityNotImplemented(
        "engine.ledger.query",
        detail="managed ledger is unavailable",
        recommended_next_action="configure a ledger binding",
    )
    encoded = encode_domain_error(original)

    assert encoded is not None
    decoded = decode_domain_error(
        code=encoded.code, message=encoded.message, details=encoded.details
    )

    assert isinstance(decoded, CapabilityNotImplemented)
    assert decoded.capability == original.capability
    assert decoded.detail == original.detail
    assert decoded.recommended_next_action == original.recommended_next_action


def test_unknown_errors_are_not_exposed_as_domain_errors() -> None:
    assert encode_domain_error(RuntimeError("secret")) is None
    assert (
        decode_domain_error(
            code="ENGINE_INTERNAL_ERROR", message="redacted", details={}
        )
        is None
    )


class _Discovery:
    def discovery(self) -> dict[str, object]:
        return {"base_url": "http://daemon", "token": "secret"}


def _async_client_for(error: dict[str, object]):
    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            del timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, _url: str, *, json: dict, headers: dict) -> httpx.Response:
            del headers
            return httpx.Response(
                200,
                json={
                    "protocol_version": "1",
                    "request_id": json["request_id"],
                    "ok": False,
                    "error": error,
                },
            )

    return FakeAsyncClient


@pytest.mark.asyncio
async def test_transport_reconstructs_domain_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = DaemonRpcTransport(tmp_path)
    transport.daemon = cast(Daemon, _Discovery())
    monkeypatch.setattr(
        daemon_client.httpx,
        "AsyncClient",
        _async_client_for(
            {
                "code": "ENGINE_CAPABILITY_NOT_IMPLEMENTED",
                "message": "Capability not implemented: engine.sources.register_repo",
                "details": {
                    "capability": "engine.sources.register_repo",
                    "detail": "upgrade required",
                    "recommended_next_action": "restart the daemon",
                },
                "retryable": False,
            }
        ),
    )

    with pytest.raises(CapabilityNotImplemented) as caught:
        await transport.call(
            "engine.sources.register_repo",
            RegisterRepoSourceRequest(pot_id="pot-1", location="owner/repo"),
        )

    assert caught.value.detail == "upgrade required"
    assert caught.value.recommended_next_action == "restart the daemon"


@pytest.mark.asyncio
async def test_old_daemon_method_not_found_recommends_restart(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = DaemonRpcTransport(tmp_path)
    transport.daemon = cast(Daemon, _Discovery())
    monkeypatch.setattr(
        daemon_client.httpx,
        "AsyncClient",
        _async_client_for(
            {
                "code": "RPC_METHOD_NOT_FOUND",
                "message": "Unknown engine RPC method.",
                "details": {},
                "retryable": False,
            }
        ),
    )

    with pytest.raises(DaemonRpcFailure) as caught:
        await transport.call(
            "engine.sources.register_repo",
            RegisterRepoSourceRequest(pot_id="pot-1", location="owner/repo"),
        )

    assert caught.value.recommended_command == "potpie daemon restart"
