"""Canonical authenticated foreground daemon runtime foundation."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import os
import secrets
import signal
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from aiohttp import web

from potpie.runtime.clients import ClientOutcome
from potpie.runtime.codec import decode_request, encode_response
from potpie.runtime.coordinator import OperationCoordinator
from potpie.runtime.operations import (
    operation_capabilities,
    operation_catalog_fingerprint,
)
from potpie.runtime.ownership import RuntimeOwnershipLock
from potpie.runtime.protocol import (
    PROTOCOL_MAX_VERSION,
    PROTOCOL_MIN_VERSION,
    PROTOCOL_VERSION,
    DaemonInternalError,
    DaemonStatusRequest,
    DaemonStatusResult,
    EngineOperationRequest,
    FailureResponse,
    HandshakeRequest,
    HandshakeResult,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolTransportError,
    RuntimeBoundaryError,
    ShutdownRequest,
    ShutdownResult,
    SuccessResponse,
)
from potpie.runtime.resource_manager import AuthenticationError
from potpie.runtime.transport import RuntimeEndpoint
from potpie_context_engine import Failure, Success


@dataclass(frozen=True, slots=True)
class AuthenticatedDaemonCaller:
    actor_id: str = "local-daemon-owner"
    authentication_scheme: str = "local_bearer"


class EngineOperationHandler(Protocol):
    async def handle(
        self,
        request: EngineOperationRequest,
        *,
        authentication: object,
    ) -> ClientOutcome: ...


ShutdownResources = Callable[[], Awaitable[object]]
AfterOwnershipAcquired = Callable[[RuntimeOwnershipLock], None]
BeforeOwnershipRelease = Callable[[RuntimeOwnershipLock], None]


def generate_bearer_token() -> str:
    """Generate one 256-bit local bearer secret for a daemon boot."""

    return secrets.token_urlsafe(32)


class CanonicalDaemonRuntime:
    """One foreground typed runtime bound to an explicit local endpoint."""

    def __init__(
        self,
        *,
        endpoint: RuntimeEndpoint,
        bearer_token: str,
        operation_handler: EngineOperationHandler,
        ownership_lock_path: Path,
        instance_id: str | None = None,
        shutdown_resources: ShutdownResources | None = None,
        after_ownership_acquired: AfterOwnershipAcquired | None = None,
        before_ownership_release: BeforeOwnershipRelease | None = None,
        coordinator: OperationCoordinator,
        backend_profile: str = "unknown",
        ui_url: str = "http://127.0.0.1",
    ) -> None:
        if len(bearer_token.encode()) < 32:
            raise ValueError("daemon bearer token must contain at least 256 bits")
        self.endpoint = endpoint
        self.instance_id = instance_id or str(uuid4())
        self._bearer_token = bearer_token
        self._operation_handler = operation_handler
        self._ownership = RuntimeOwnershipLock(ownership_lock_path)
        self._shutdown_resources = shutdown_resources
        self._after_ownership_acquired = after_ownership_acquired
        self._before_ownership_release = before_ownership_release
        self._coordinator = coordinator
        self._backend_profile = backend_profile
        self._ui_url = ui_url
        self._state = "starting"
        self._runner: web.AppRunner | None = None
        self._shutdown_requested = asyncio.Event()
        self._state_condition = asyncio.Condition()
        self._active_operations = 0
        self._accepting_operations = False
        self._compatibility_secret = secrets.token_bytes(32)
        self._stop_lock = asyncio.Lock()

    @property
    def lifecycle_state(self) -> str:
        return self._state

    @property
    def pid(self) -> int:
        return os.getpid()

    async def start(self) -> None:
        if self._runner is not None or self._state != "starting":
            raise RuntimeError("canonical daemon runtime can start only once")
        ownership = self._ownership.acquire()
        if isinstance(ownership, Failure):
            self._state = "failed"
            raise RuntimeError(ownership.error.code)
        try:
            if self._after_ownership_acquired is not None:
                self._after_ownership_acquired(self._ownership)
        except Exception:
            self._state = "failed"
            self._ownership.release()
            raise
        app = web.Application()
        app.router.add_post("/v1/operations", self._handle_http)
        runner = web.AppRunner(app, access_log=None)
        self._runner = runner
        socket_was_absent = False
        try:
            await runner.setup()
            if self.endpoint.kind == "uds":
                socket_path = Path(self.endpoint.address)
                socket_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                socket_path.parent.chmod(0o700)
                if socket_path.exists():
                    raise FileExistsError(
                        f"daemon socket already exists: {socket_path}"
                    )
                socket_was_absent = True
                site: web.BaseSite = web.UnixSite(runner, str(socket_path))
            else:
                site = web.TCPSite(
                    runner,
                    host=self.endpoint.address,
                    port=self.endpoint.port,
                )
            await site.start()
            if self.endpoint.kind == "uds":
                Path(self.endpoint.address).chmod(0o600)
            async with self._state_condition:
                self._state = "ready"
                self._accepting_operations = True
                self._state_condition.notify_all()
        except Exception:
            self._state = "failed"
            await runner.cleanup()
            if self.endpoint.kind == "uds" and socket_was_absent:
                with contextlib.suppress(FileNotFoundError):
                    Path(self.endpoint.address).unlink()
            self._ownership.release()
            self._runner = None
            raise

    async def serve_until_shutdown(self) -> None:
        if self._runner is None:
            await self.start()
        await self._shutdown_requested.wait()
        await self.stop()

    async def request_shutdown(self) -> None:
        async with self._state_condition:
            if self._state in {"stopped", "failed"}:
                return
            self._state = "draining"
            self._accepting_operations = False
            self._state_condition.notify_all()
        self._shutdown_requested.set()

    async def stop(self) -> None:
        async with self._stop_lock:
            if self._state == "stopped":
                return
            await self.request_shutdown()
            runner, self._runner = self._runner, None
            if runner is not None:
                await runner.cleanup()
            async with self._state_condition:
                await self._state_condition.wait_for(
                    lambda: self._active_operations == 0
                )
            cleanup_failed = False
            if self._shutdown_resources is not None:
                try:
                    cleanup = await self._shutdown_resources()
                    cleanup_failed = isinstance(cleanup, Failure)
                except Exception:
                    self._state = "failed"
                    cleanup_failed = True
            if self.endpoint.kind == "uds":
                socket_path = Path(self.endpoint.address)
                with contextlib.suppress(FileNotFoundError):
                    socket_path.unlink()
            if self._before_ownership_release is not None and self._ownership.is_held:
                try:
                    self._before_ownership_release(self._ownership)
                except Exception:
                    self._state = "failed"
                    cleanup_failed = True
            self._ownership.release()
            self._state = "stopped"
            if cleanup_failed:
                raise RuntimeError("daemon resource shutdown failed")

    async def _handle_http(self, request: web.Request) -> web.Response:
        try:
            document = await request.json()
        except Exception:
            document = None
        protocol_version, request_id = _correlation_identity(document)
        if not self._authenticated(request.headers):
            return _json_response(
                FailureResponse(
                    protocol_version=protocol_version,
                    request_id=request_id,
                    outcome=Failure(
                        AuthenticationError(
                            code="daemon_authentication_failed",
                            message="daemon authentication failed",
                        )
                    ),
                ),
                status=401,
            )
        decoded = decode_request(document)
        if isinstance(decoded, Failure):
            return _json_response(
                FailureResponse(
                    protocol_version=protocol_version,
                    request_id=request_id,
                    outcome=decoded,
                ),
                status=400,
            )
        typed_request = decoded.value
        if (
            not PROTOCOL_MIN_VERSION
            <= typed_request.protocol_version
            <= PROTOCOL_MAX_VERSION
        ):
            return _json_response(
                _failure_response(
                    typed_request,
                    ProtocolError(
                        code="protocol_version_incompatible",
                        message="request protocol version is incompatible",
                        details={
                            "supported_min": PROTOCOL_MIN_VERSION,
                            "supported_max": PROTOCOL_MAX_VERSION,
                        },
                        recommended_next_action="install compatible Potpie versions",
                    ),
                ),
                status=409,
            )
        try:
            response, status = await self._execute(typed_request)
        except Exception:
            response = _failure_response(
                typed_request,
                DaemonInternalError(
                    code="daemon_internal_failure",
                    message="the daemon failed to process the request",
                    details={"request_id": typed_request.request_id},
                    recommended_next_action="inspect daemon logs",
                ),
            )
            status = 500
        try:
            return _json_response(response, status=status)
        except Exception:
            return _json_response(
                _failure_response(
                    typed_request,
                    DaemonInternalError(
                        code="daemon_response_encoding_failed",
                        message="the daemon could not encode the operation response",
                        details={"request_id": typed_request.request_id},
                        recommended_next_action="inspect daemon logs",
                    ),
                ),
                status=500,
            )

    async def _execute(self, request: ProtocolRequest) -> tuple[ProtocolResponse, int]:
        if isinstance(request, HandshakeRequest):
            return self._handshake(request)
        if not self._valid_compatibility_ticket(request.compatibility_ticket):
            return (
                _failure_response(
                    request,
                    ProtocolError(
                        code="compatibility_ticket_invalid",
                        message=(
                            "a ticket from a compatible handshake is required before "
                            "this operation"
                        ),
                        recommended_next_action="perform an authenticated handshake",
                        retry_posture="safe",
                    ),
                ),
                409,
            )
        if isinstance(request, ShutdownRequest):
            async with self._coordinator.lifecycle_control():
                await self.request_shutdown()
                return (
                    SuccessResponse(
                        protocol_version=request.protocol_version,
                        request_id=request.request_id,
                        outcome=Success(ShutdownResult(accepted=True)),
                    ),
                    200,
                )
        if isinstance(request, DaemonStatusRequest):
            async with self._coordinator.lifecycle_control():
                return (
                    SuccessResponse(
                        protocol_version=request.protocol_version,
                        request_id=request.request_id,
                        outcome=Success(
                            DaemonStatusResult(
                                instance_id=self.instance_id,
                                pid=self.pid,
                                lifecycle_state=self._state,  # type: ignore[arg-type]
                                backend_profile=self._backend_profile,
                                ui_url=self._ui_url,
                            )
                        ),
                    ),
                    200,
                )
        async with self._state_condition:
            if not self._accepting_operations:
                return (
                    _failure_response(
                        request,
                        ProtocolTransportError(
                            code="daemon_draining",
                            message="daemon is not accepting new operations",
                            retry_posture="safe",
                        ),
                    ),
                    503,
                )
            self._active_operations += 1
        try:
            outcome = await self._operation_handler.handle(
                request,
                authentication=AuthenticatedDaemonCaller(),
            )
        finally:
            async with self._state_condition:
                self._active_operations -= 1
                self._state_condition.notify_all()
        if isinstance(outcome, Success):
            return (
                SuccessResponse(
                    protocol_version=request.protocol_version,
                    request_id=request.request_id,
                    outcome=outcome,
                ),
                200,
            )
        return _failure_response(request, outcome.error), _status_for(outcome.error)

    def _handshake(self, request: HandshakeRequest) -> tuple[ProtocolResponse, int]:
        payload = request.payload
        if not (
            payload.client_protocol_min <= PROTOCOL_MAX_VERSION
            and PROTOCOL_MIN_VERSION <= payload.client_protocol_max
        ):
            return (
                _failure_response(
                    request,
                    ProtocolError(
                        code="protocol_version_incompatible",
                        message="client and daemon protocol ranges do not overlap",
                        recommended_next_action="install compatible Potpie versions",
                    ),
                ),
                409,
            )
        if (
            payload.expected_instance_id is not None
            and payload.expected_instance_id != self.instance_id
        ):
            return (
                _failure_response(
                    request,
                    ProtocolError(
                        code="daemon_instance_mismatch",
                        message="daemon instance identity does not match the request",
                        recommended_next_action="refresh discovery and reconnect",
                    ),
                ),
                409,
            )
        catalog_fingerprint = operation_catalog_fingerprint()
        if payload.client_operation_catalog_fingerprint != catalog_fingerprint:
            return (
                _failure_response(
                    request,
                    ProtocolError(
                        code="operation_catalog_mismatch",
                        message="client and daemon operation catalogs do not match",
                        recommended_next_action=(
                            "restart with a compatible Potpie version"
                        ),
                    ),
                ),
                409,
            )
        return (
            SuccessResponse(
                protocol_version=request.protocol_version,
                request_id=request.request_id,
                outcome=Success(
                    HandshakeResult(
                        protocol_min=PROTOCOL_MIN_VERSION,
                        protocol_max=PROTOCOL_MAX_VERSION,
                        instance_id=self.instance_id,
                        lifecycle_state=self._state,  # type: ignore[arg-type]
                        capabilities=operation_capabilities(),
                        operation_catalog_fingerprint=catalog_fingerprint,
                        compatibility_ticket=self._issue_compatibility_ticket(),
                    )
                ),
            ),
            200,
        )

    def _issue_compatibility_ticket(self) -> str:
        nonce = secrets.token_urlsafe(24)
        signature = hmac.new(
            self._compatibility_secret,
            self._compatibility_ticket_message(nonce),
            "sha256",
        ).hexdigest()
        return f"{nonce}.{signature}"

    def _valid_compatibility_ticket(self, ticket: str | None) -> bool:
        if not ticket:
            return False
        try:
            nonce, supplied_signature = ticket.rsplit(".", 1)
        except ValueError:
            return False
        expected_signature = hmac.new(
            self._compatibility_secret,
            self._compatibility_ticket_message(nonce),
            "sha256",
        ).hexdigest()
        return hmac.compare_digest(supplied_signature, expected_signature)

    def _compatibility_ticket_message(self, nonce: str) -> bytes:
        return (
            f"{self.instance_id}:{PROTOCOL_VERSION}:"
            f"{operation_catalog_fingerprint()}:{nonce}"
        ).encode()

    def _authenticated(self, headers: Mapping[str, str]) -> bool:
        supplied = headers.get("Authorization", "")
        expected = f"Bearer {self._bearer_token}"
        return hmac.compare_digest(supplied.encode(), expected.encode())


async def run_foreground(runtime: CanonicalDaemonRuntime) -> None:
    """Run one supplied runtime until typed control or a termination signal."""

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(
                signum,
                lambda: asyncio.create_task(runtime.request_shutdown()),
            )
    try:
        await runtime.serve_until_shutdown()
    finally:
        await runtime.stop()


def _correlation_identity(document: object) -> tuple[int, str]:
    if not isinstance(document, Mapping):
        return PROTOCOL_VERSION, str(uuid4())
    protocol_version = document.get("protocol_version")
    if not isinstance(protocol_version, int) or isinstance(protocol_version, bool):
        protocol_version = PROTOCOL_VERSION
    request_id = document.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        request_id = str(uuid4())
    return protocol_version, request_id


def _failure_response(
    request: ProtocolRequest, error: RuntimeBoundaryError
) -> FailureResponse:
    return FailureResponse(
        protocol_version=request.protocol_version,
        request_id=request.request_id,
        outcome=Failure(error),
    )


def _json_response(response: ProtocolResponse, *, status: int) -> web.Response:
    return web.json_response(encode_response(response), status=status)


def _status_for(error: RuntimeBoundaryError) -> int:
    return {
        "authentication": 401,
        "authorization": 403,
        "protocol": 400,
        "protocol_transport": 503,
        "daemon_internal": 500,
    }.get(error.category, 200)


__all__ = [
    "AuthenticatedDaemonCaller",
    "CanonicalDaemonRuntime",
    "EngineOperationHandler",
    "generate_bearer_token",
    "run_foreground",
]
