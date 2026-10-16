"""A daemon from another build stays observable and stoppable.

A daemon whose operation catalog differs from the client's answers the
handshake with a ticket scoped to daemon status and shutdown. The client keeps
that ticket for those two operations and never treats it as readiness.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

from dataclasses import fields

import pytest

from potpie.runtime import (
    PROTOCOL_VERSION,
    DaemonBuild,
    DaemonControlClient,
    DaemonControlOperation,
    DaemonStatusPayload,
    DaemonStatusRequest,
    DaemonStatusResult,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    ShutdownPayload,
    ShutdownRequest,
    ShutdownResult,
    SuccessResponse,
    operation_capabilities,
    operation_catalog_fingerprint,
)
from potpie.runtime.protocol import CONTROL_TICKET_OPERATIONS
from potpie_context_engine import Failure, Success

pytestmark = pytest.mark.unit

_OTHER_CATALOG = "0" * 64


def _handshake_result(**changes: object) -> HandshakeResult:
    values: dict[str, object] = {
        "protocol_min": PROTOCOL_VERSION,
        "protocol_max": PROTOCOL_VERSION,
        "instance_id": "instance-1",
        "lifecycle_state": "ready",
        "capabilities": operation_capabilities(),
        "operation_catalog_fingerprint": operation_catalog_fingerprint(),
        "compatibility_ticket": "full-ticket",
    }
    values.update(changes)
    return HandshakeResult(**values)  # type: ignore[arg-type]


class _Daemon:
    """Answers the handshake with ``handshake`` and serves status and shutdown."""

    def __init__(self, handshake: HandshakeResult) -> None:
        self.handshake = handshake
        self.requests: list[object] = []

    async def send(self, request):
        self.requests.append(request)
        value: object
        if isinstance(request, HandshakeRequest):
            value = self.handshake
        elif isinstance(request, DaemonStatusRequest):
            value = DaemonStatusResult(
                instance_id="instance-1",
                pid=4242,
                lifecycle_state="ready",
                backend_profile="embedded",
                ui_url="http://127.0.0.1:8765",
            )
        elif isinstance(request, ShutdownRequest):
            value = ShutdownResult(accepted=True)
        else:  # pragma: no cover - the control client sends nothing else
            raise AssertionError(f"unexpected request {request!r}")
        return SuccessResponse(
            protocol_version=request.protocol_version,
            request_id=request.request_id,
            outcome=Success(value),
        )


def _client(daemon: _Daemon) -> DaemonControlClient:
    return DaemonControlClient(transport=daemon, expected_instance_id="instance-1")


def _sent(daemon: _Daemon, kind: type) -> list:
    return [request for request in daemon.requests if isinstance(request, kind)]


@pytest.mark.anyio
async def test_another_catalog_is_not_readiness_but_keeps_a_control_ticket() -> None:
    daemon = _Daemon(
        _handshake_result(
            operation_catalog_fingerprint=_OTHER_CATALOG,
            compatibility_ticket="control-ticket",
        )
    )
    client = _client(daemon)

    readiness = await client.handshake()
    status = await client.status()
    shutdown = await client.shutdown(reason="catalog_change")

    assert isinstance(readiness, Failure)
    assert readiness.error.code == "operation_catalog_mismatch"
    assert client.handshake_result is None
    assert client.catalog_compatible is False
    assert isinstance(status, Success)
    assert isinstance(shutdown, Success)
    assert [request.compatibility_ticket for request in daemon.requests[1:]] == [
        "control-ticket",
        "control-ticket",
    ]


@pytest.mark.anyio
async def test_control_handshake_reaches_a_daemon_from_another_build() -> None:
    daemon = _Daemon(
        _handshake_result(
            operation_catalog_fingerprint=_OTHER_CATALOG,
            capabilities=tuple(
                sorted(
                    {DaemonControlOperation.HANDSHAKE.value}
                    | {operation.value for operation in CONTROL_TICKET_OPERATIONS}
                )
            ),
        )
    )
    client = _client(daemon)

    handshake = await client.control_handshake()
    again = await client.control_handshake()

    assert isinstance(handshake, Success)
    assert again == handshake
    assert len(_sent(daemon, HandshakeRequest)) == 1
    assert client.catalog_compatible is False


@pytest.mark.anyio
async def test_control_handshake_with_this_catalog_is_full_readiness() -> None:
    daemon = _Daemon(_handshake_result())
    client = _client(daemon)

    handshake = await client.control_handshake()

    assert isinstance(handshake, Success)
    assert client.handshake_result == _handshake_result()
    assert client.catalog_compatible is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"instance_id": "other"}, "daemon_instance_mismatch"),
        ({"lifecycle_state": "draining"}, "daemon_not_ready"),
        (
            {
                "protocol_min": PROTOCOL_VERSION + 1,
                "protocol_max": PROTOCOL_VERSION + 1,
            },
            "protocol_version_incompatible",
        ),
        ({"compatibility_ticket": ""}, "compatibility_ticket_missing"),
        (
            {"capabilities": (DaemonControlOperation.HANDSHAKE.value,)},
            "daemon_capability_missing",
        ),
    ],
)
async def test_control_handshake_still_requires_identity_protocol_and_control(
    changes: dict[str, object], expected_code: str
) -> None:
    daemon = _Daemon(
        _handshake_result(operation_catalog_fingerprint=_OTHER_CATALOG, **changes)
    )
    client = _client(daemon)

    handshake = await client.control_handshake()
    status = await client.status()

    assert isinstance(handshake, Failure)
    assert handshake.error.code == expected_code
    assert client.control_result is None
    assert isinstance(status, Failure)
    assert status.error.code == "handshake_required"
    assert _sent(daemon, DaemonStatusRequest) == []


def test_control_ticket_reaches_only_status_and_shutdown() -> None:
    assert CONTROL_TICKET_OPERATIONS == frozenset(
        {DaemonControlOperation.STATUS, DaemonControlOperation.SHUTDOWN}
    )


def test_daemon_control_wire_shape_changes_only_with_the_protocol_version() -> None:
    """Control tickets cross operation-catalog changes, so the handshake,
    status, and shutdown wire shapes belong to the protocol version: changing
    any of them must come with a ``PROTOCOL_VERSION`` change, and this pin
    with it.
    """
    wire_types = (
        HandshakePayload,
        HandshakeResult,
        DaemonStatusPayload,
        DaemonStatusResult,
        DaemonBuild,
        ShutdownPayload,
        ShutdownResult,
    )

    shapes = {
        wire_type.__name__: tuple(field.name for field in fields(wire_type))
        for wire_type in wire_types
    }

    assert PROTOCOL_VERSION == 2
    assert shapes == {
        "HandshakePayload": (
            "client_protocol_min",
            "client_protocol_max",
            "expected_instance_id",
            "client_operation_catalog_fingerprint",
        ),
        "HandshakeResult": (
            "protocol_min",
            "protocol_max",
            "instance_id",
            "lifecycle_state",
            "capabilities",
            "operation_catalog_fingerprint",
            "compatibility_ticket",
        ),
        "DaemonStatusPayload": (),
        "DaemonStatusResult": (
            "instance_id",
            "pid",
            "lifecycle_state",
            "backend_profile",
            "ui_url",
            "version",
            "build",
        ),
        "DaemonBuild": ("rev", "dirty", "built_at"),
        "ShutdownPayload": ("reason",),
        "ShutdownResult": ("accepted",),
    }
