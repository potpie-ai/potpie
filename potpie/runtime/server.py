"""Canonical authenticated foreground daemon runtime foundation.

The local daemon is aiohttp wiring around the transport-neutral
:class:`~potpie.runtime.endpoint.OperationEndpoint`: a per-boot bearer
credential, per-boot compatibility tickets (catalog lock-step), a UDS or
loopback TCP listener and an explicit request size limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from aiohttp import web

from potpie.runtime.clients import ClientOutcome
from potpie.runtime.coordinator import OperationCoordinator
from potpie.runtime.endpoint import (
    DEFAULT_MAX_REQUEST_BYTES,
    BearerTokenAuthenticator,
    HostIdentity,
    OperationEndpoint,
    PerBootTicketCompatibility,
)
from potpie.runtime.operations import operation_catalog_fingerprint
from potpie.runtime.ownership import RuntimeOwnershipLock
from potpie.runtime.protocol import DaemonBuild, EngineOperationRequest
from potpie.runtime.resource_manager import AuthenticatedActor
from potpie.runtime.transport import RuntimeEndpoint
from potpie_context_engine import Failure


@dataclass(frozen=True, slots=True)
class AuthenticatedDaemonCaller:
    actor_id: str = "local-daemon-owner"
    authentication_scheme: str = "local_bearer"


#: The actor every request with the daemon's per-boot credential acts as.
LOCAL_DAEMON_ACTOR = AuthenticatedActor(
    actor_id=AuthenticatedDaemonCaller.actor_id,
    attributes={
        "authentication_scheme": AuthenticatedDaemonCaller.authentication_scheme
    },
)


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


def _catalog_fingerprint() -> str:
    """This build's catalog fingerprint, looked up when it is needed."""

    return operation_catalog_fingerprint()


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
        version: str | None = None,
        build: DaemonBuild | None = None,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        if len(bearer_token.encode()) < 32:
            raise ValueError("daemon bearer token must contain at least 256 bits")
        self.endpoint = endpoint
        self.instance_id = instance_id or str(uuid4())
        self._ownership = RuntimeOwnershipLock(ownership_lock_path)
        self._shutdown_resources = shutdown_resources
        self._after_ownership_acquired = after_ownership_acquired
        self._before_ownership_release = before_ownership_release
        self._max_request_bytes = max_request_bytes
        self._operations = OperationEndpoint(
            authenticator=BearerTokenAuthenticator(
                bearer_token,
                actor=LOCAL_DAEMON_ACTOR,
                failure_code="daemon_authentication_failed",
            ),
            compatibility=PerBootTicketCompatibility(
                instance_id=self.instance_id,
                catalog_fingerprint=_catalog_fingerprint,
            ),
            engine_handler=operation_handler,
            identity=HostIdentity(
                host_id=self.instance_id,
                version=version,
                build=build,
                backend_profile=backend_profile,
                ui_url=ui_url,
            ),
            max_request_bytes=max_request_bytes,
            coordinator=coordinator,
            on_shutdown=self.request_shutdown,
            lifecycle_state="starting",
            catalog_fingerprint=_catalog_fingerprint,
        )
        self._runner: web.AppRunner | None = None
        self._shutdown_requested = asyncio.Event()
        self._stop_lock = asyncio.Lock()

    @property
    def operation_endpoint(self) -> OperationEndpoint:
        return self._operations

    @property
    def lifecycle_state(self) -> str:
        return self._operations.lifecycle_state

    @property
    def pid(self) -> int:
        return os.getpid()

    async def start(self) -> None:
        if self._runner is not None or self.lifecycle_state != "starting":
            raise RuntimeError("canonical daemon runtime can start only once")
        ownership = self._ownership.acquire()
        if isinstance(ownership, Failure):
            self._operations.set_lifecycle_state("failed")
            raise RuntimeError(ownership.error.code)
        try:
            if self._after_ownership_acquired is not None:
                self._after_ownership_acquired(self._ownership)
        except Exception:
            self._operations.set_lifecycle_state("failed")
            self._ownership.release()
            raise
        # An explicit limit: aiohttp's own default (1 MiB) refused large
        # resource imports as malformed envelopes.
        app = web.Application(client_max_size=self._max_request_bytes)
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
            await self._operations.mark_ready()
        except Exception:
            self._operations.set_lifecycle_state("failed")
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
        if not await self._operations.begin_drain():
            return
        self._shutdown_requested.set()

    async def stop(self) -> None:
        async with self._stop_lock:
            if self.lifecycle_state == "stopped":
                return
            await self.request_shutdown()
            runner, self._runner = self._runner, None
            if runner is not None:
                await runner.cleanup()
            await self._operations.wait_idle()
            cleanup_failed = False
            if self._shutdown_resources is not None:
                try:
                    cleanup = await self._shutdown_resources()
                    cleanup_failed = isinstance(cleanup, Failure)
                except Exception:
                    self._operations.set_lifecycle_state("failed")
                    cleanup_failed = True
            if self.endpoint.kind == "uds":
                socket_path = Path(self.endpoint.address)
                with contextlib.suppress(FileNotFoundError):
                    socket_path.unlink()
            if self._before_ownership_release is not None and self._ownership.is_held:
                try:
                    self._before_ownership_release(self._ownership)
                except Exception:
                    self._operations.set_lifecycle_state("failed")
                    cleanup_failed = True
            self._ownership.release()
            self._operations.set_lifecycle_state("stopped")
            if cleanup_failed:
                raise RuntimeError("daemon resource shutdown failed")

    async def _handle_http(self, request: web.Request) -> web.Response:
        if (
            request.content_length is not None
            and request.content_length > self._max_request_bytes
        ):
            response = self._operations.refuse_oversized()
        else:
            try:
                body = await request.read()
            except web.HTTPRequestEntityTooLarge:
                response = self._operations.refuse_oversized()
            else:
                response = await self._operations.handle(body, request.headers)
        return web.Response(
            status=response.status,
            body=response.body,
            headers=dict(response.headers),
        )


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


__all__ = [
    "LOCAL_DAEMON_ACTOR",
    "AuthenticatedDaemonCaller",
    "CanonicalDaemonRuntime",
    "EngineOperationHandler",
    "generate_bearer_token",
    "run_foreground",
]
