"""What an embedding host's contract tests import to speak the wire.

**Provisional**, like :mod:`potpie.runtime.hosting`: names may still move
while the managed-host contract settles.

A host serves requests; its tests also have to send them. This module holds
the client half they need and nothing a serving host does: the request
envelopes (engine, ``host.*`` / ``ext.*`` and daemon control), the destructive
confirmation an envelope carries, and the codec that turns an envelope into
the JSON document a remote client posts and a reply back into a typed
response::

    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="request-1",
        operation=EngineOperation.RESET_CONTEXT,
        selector=selector,
        payload=ResetContextRequest(),
        destructive_intent=DestructiveIntent(
            confirmed=True,
            operation="reset_context",
            selector=selector,
            request_id="request-1",
        ),
    )
    body = json.dumps(encode_request(request)).encode()
    reply = await endpoint.handle(body, headers)
    decoded = decode_response(json.loads(reply.body), request=request)

Engine payload types (``ResolveRequest`` and the rest) come from
``potpie_context_engine.api``; selectors, operation names and the host
catalog's request types from :mod:`potpie.runtime.hosting`.
"""

from __future__ import annotations

from potpie.runtime.codec import (
    REQUEST_FIELDS_UNSUPPORTED,
    decode_request,
    decode_response,
    encode_request,
)
from potpie.runtime.protocol import (
    PROTOCOL_VERSION,
    DaemonStatusPayload,
    DaemonStatusRequest,
    DaemonStatusResult,
    EngineOperationRequest,
    FailureResponse,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    HostOperationRequest,
    ProtocolRequest,
    ProtocolResponse,
    ShutdownPayload,
    ShutdownRequest,
    SuccessResponse,
)
from potpie.runtime.resource_manager import DestructiveIntent

__all__ = [
    "PROTOCOL_VERSION",
    "REQUEST_FIELDS_UNSUPPORTED",
    "DaemonStatusPayload",
    "DaemonStatusRequest",
    "DaemonStatusResult",
    "DestructiveIntent",
    "EngineOperationRequest",
    "FailureResponse",
    "HandshakePayload",
    "HandshakeRequest",
    "HandshakeResult",
    "HostOperationRequest",
    "ProtocolRequest",
    "ProtocolResponse",
    "ShutdownPayload",
    "ShutdownRequest",
    "SuccessResponse",
    "decode_request",
    "decode_response",
    "encode_request",
]
