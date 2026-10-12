"""Typed client for a managed host: HTTPS, one bearer key, no lock-step.

A managed host serves the same typed envelope as the local daemon at
``POST <base>/v1/operations``. Unlike the local daemon it keeps no per-boot
state: no handshake is required, no compatibility ticket is sent, and a
catalog fingerprint that differs from this client's is not a refusal. Version
skew is handled per call instead:

* requests leave default-valued fields off the wire, so a call that uses
  nothing new reads exactly like an older client's;
* a host that does not serve an operation or a field says so precisely, and
  this client turns that into ``host_outdated`` naming the host, the
  operation and the fields (:func:`host_outdated_error`); a caller may retry
  without an optional field with :func:`without_fields`;
* results decode leniently, so fields a newer host adds are ignored.

URL rules: ``https://`` is required, except plain ``http://`` for a loopback
host or for the one opted-in insecure ``host[:port]``. They are checked when an
endpoint is built and again for every transport, and a key sent over plain
http to a non-loopback host prints a one-line warning on stderr.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from typing import Any, TypeVar, overload
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from potpie.config import host_registry
from potpie.pots.contracts import (
    PotAggregateStatus,
    PotInfo,
    PotRepoSource,
    SourceInfo,
)
from potpie.pots.directory import is_remote_repo_key
from potpie.runtime.clients import (
    ClientOutcome,
    DaemonTransport,
    DestructiveConfirmation,
    EngineClient,
    RequestIdFactory,
    _build_operation_request,
    _send_protocol_request,
)
from potpie.runtime.codec import decode_response, encode_request
from potpie.runtime.host_operations import (
    HOST_OPERATION_CATALOG,
    GrantRevokeRequest,
    GrantSetRequest,
    GrantListRequest,
    HostOperation,
    PotCreateRequest,
    PotGrantInfo,
    PotListRequest,
    PotListResult,
    PotStatusRequest,
    PotUseRequest,
    RepoDefaultClearRequest,
    RepoDefaultSetRequest,
    SourceAddRequest,
    SourceListRequest,
    WhoamiRequest,
    WhoamiResult,
)
from potpie.runtime.operations import EngineOperation, SafetyClass
from potpie.runtime.protocol import (
    PROTOCOL_MAX_VERSION,
    PROTOCOL_MIN_VERSION,
    PROTOCOL_VERSION,
    DaemonStatusPayload,
    DaemonStatusRequest,
    DaemonStatusResult,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    HostOperationRequest,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    RuntimeBoundaryError,
    TransportFailure,
)
from potpie.runtime.resource_manager import ContextSelector, DestructiveIntent
from potpie_context_engine import Failure, Success
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.ports.agent_context import (
    RecordRequest as AgentRecordRequest,
)
from potpie_context_engine.core.ports.agent_context import (
    ResolveRequest as AgentResolveRequest,
)
from potpie_context_engine.core.ports.agent_context import (
    SearchRequest as AgentSearchRequest,
)
from potpie_context_engine.requests import (
    EngineRequest,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from potpie_context_engine.results import result_from_payload

HOST_OUTDATED = "host_outdated"
DEFAULT_TIMEOUT_S = 30.0
#: How long :class:`ManagedPotDirectory` serves a pot listing from its cache.
POT_LISTING_MAX_AGE_S = 5.0
_CLOSE_ON_OWN_LOOP_S = 5.0
_OPERATIONS_PATH = "v1/operations"
_OUTDATED_SOURCES = frozenset({"operation_unknown", "request_fields_unsupported"})

ResultT = TypeVar("ResultT")
RequestT = TypeVar("RequestT")


# --- addresses ---------------------------------------------------------------------


def _normalize_managed_url(url: str, allow_insecure_host: str | None) -> str:
    """The registry's URL rules, the single definition of what is acceptable."""

    return host_registry.normalize_managed_url(
        url, allow_insecure_host=allow_insecure_host
    )


def _warn_insecure_transport(base_url: str) -> None:
    host_registry.warn_insecure_transport(base_url)


@dataclass(frozen=True, slots=True)
class ManagedHostEndpoint:
    """A validated managed host address and the key sent to it.

    ``base_url`` may carry a path prefix (``https://your-host.example/graph``);
    the operation URL joins it to ``v1/operations`` with exactly one ``/``. An
    empty ``token`` is accepted only for a loopback host, and then no
    ``Authorization`` header is sent at all.
    """

    base_url: str
    token: str | None = field(default=None, repr=False)
    allow_insecure_host: str | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_managed_url(self.base_url, self.allow_insecure_host)
        object.__setattr__(self, "base_url", normalized)
        token = (self.token or "").strip() or None
        object.__setattr__(self, "token", token)
        if token is None and not host_registry.is_loopback_host(
            urlsplit(normalized).hostname
        ):
            raise ValueError(
                f"the managed host {normalized} has no token, and an empty token is "
                "accepted only for a loopback host"
            )

    @property
    def operations_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{_OPERATIONS_PATH}"

    @property
    def insecure(self) -> bool:
        """Plain http to a non-loopback host (the opted-in exception)."""

        parts = urlsplit(self.base_url)
        return parts.scheme == "http" and not host_registry.is_loopback_host(
            parts.hostname
        )

    @property
    def label(self) -> str:
        """``Managed (host[:port]/prefix)``, for messages."""

        trimmed = self.base_url.split("://", 1)[-1]
        return f"Managed ({trimmed})"


# --- transport --------------------------------------------------------------------


_user_agent: str | None = None


def _default_user_agent() -> str:
    global _user_agent
    if _user_agent is None:
        try:
            from potpie import build_info

            version = build_info.distribution_version() or "unknown"
            rev = build_info.short_rev(build_info.build_stamp().get("rev"))
        except Exception:  # noqa: BLE001 - an identity header is best effort
            version, rev = "unknown", None
        _user_agent = f"potpie/{version} ({rev or 'unknown'})"
    return _user_agent


class HttpsHostTransport:
    """Send typed requests to one managed host over pooled HTTPS clients.

    Satisfies ``DaemonTransport.send``. An ``httpx.AsyncClient``'s connections
    belong to the event loop that opened them, so the transport keeps one
    keep-alive client per event loop it is used from, and loops on different
    threads never share or close each other's client. A client whose loop has
    closed is closed the next time the transport is used, and :meth:`aclose`
    closes every client, each on its own loop when that loop is still
    running. Transport failures raise ``TransportFailure`` with ``dispatched``
    saying whether the host may have acted.
    """

    def __init__(
        self,
        endpoint: ManagedHostEndpoint,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        user_agent: str | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("managed host timeout must be positive")
        # Checked again for every transport, so an edited configuration that
        # slipped past an earlier check still fails closed here.
        _normalize_managed_url(endpoint.base_url, endpoint.allow_insecure_host)
        self.endpoint = endpoint
        self._timeout = httpx.Timeout(timeout_s)
        self._user_agent = user_agent
        self._http_transport = http_transport
        self._lock = threading.Lock()
        self._clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}

    async def send(self, request: ProtocolRequest) -> ProtocolResponse:
        client, retired = self._client_for_running_loop()
        for loop, stale in retired:
            await _close_client(stale, loop)
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request.request_id,
            "User-Agent": self._user_agent or _default_user_agent(),
        }
        if self.endpoint.token:
            if self.endpoint.insecure:
                _warn_insecure_transport(self.endpoint.base_url)
            headers["Authorization"] = f"Bearer {self.endpoint.token}"
        try:
            response = await client.post(
                self.endpoint.operations_url,
                json=encode_request(request),
                headers=headers,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise TransportFailure(code="host_unavailable", dispatched=False) from exc
        except (
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.WriteError,
            httpx.WriteTimeout,
        ) as exc:
            raise TransportFailure(
                code="host_connection_lost", dispatched=True
            ) from exc
        except httpx.RequestError as exc:
            raise TransportFailure(
                code="host_transport_failed", dispatched=True
            ) from exc

        try:
            document = response.json()
        except ValueError as exc:
            # A route that does not exist never reached an operation handler.
            raise TransportFailure(
                code=_unanswered_code(response.status_code),
                dispatched=response.status_code != 404,
            ) from exc
        decoded = decode_response(document, request=request)
        if isinstance(decoded, Failure):
            if response.status_code == 404:
                raise TransportFailure(code="host_route_not_found", dispatched=False)
            raise TransportFailure(code=decoded.error.code, dispatched=True)
        return decoded.value

    @property
    def is_open(self) -> bool:
        """Whether a pooled client exists (nothing to close otherwise)."""
        with self._lock:
            return bool(self._clients)

    async def aclose(self) -> None:
        """Close every pooled client; the next call opens a new one."""
        with self._lock:
            clients, self._clients = list(self._clients.items()), {}
        for loop, client in clients:
            await _close_client(client, loop)

    close = aclose

    def _client_for_running_loop(
        self,
    ) -> tuple[
        httpx.AsyncClient,
        list[tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]],
    ]:
        """This loop's client, and the clients of loops that have since closed."""
        loop = asyncio.get_running_loop()
        retired: list[tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = []
        with self._lock:
            client = self._clients.get(loop)
            if client is None:
                retired = [
                    (other, stale)
                    for other, stale in self._clients.items()
                    if other.is_closed()
                ]
                for other, _ in retired:
                    del self._clients[other]
                client = self._clients[loop] = httpx.AsyncClient(
                    timeout=self._timeout,
                    transport=self._http_transport,
                    follow_redirects=False,
                )
        return client, retired


async def _close_client(
    client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop
) -> None:
    """Close ``client`` on ``loop`` when that loop still runs elsewhere, else here.

    A loop that has closed (or stops before it gets to the close) cannot
    finish closing its sockets; closing the client here still shuts its
    connections down and marks it closed, and the garbage collector releases
    what the dead loop could not.
    """
    if client.is_closed:
        return
    if loop is not asyncio.get_running_loop() and loop.is_running():
        try:
            closing = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
            await asyncio.wait_for(
                asyncio.wrap_future(closing), timeout=_CLOSE_ON_OWN_LOOP_S
            )
            return
        except (RuntimeError, TimeoutError):
            pass
    try:
        await client.aclose()
    except RuntimeError:  # its loop has closed
        pass


def _unanswered_code(status: int) -> str:
    if status == 404:
        return "host_route_not_found"
    if status in (502, 503, 504):
        return "host_unavailable"
    return "host_response_not_json"


# --- version skew -----------------------------------------------------------------


def host_outdated_error(
    error: RuntimeBoundaryError, *, host: str, operation: str
) -> RuntimeBoundaryError:
    """``operation_unknown``/``request_fields_unsupported`` as ``host_outdated``.

    A capability answer, not a fault: everything the host already supports
    keeps working. Any other error is returned unchanged.
    """

    if error.category != "protocol" or error.code not in _OUTDATED_SOURCES:
        return error
    fields_named = tuple(str(item) for item in (error.details.get("fields") or ()))
    missing = (
        ", ".join(repr(name) for name in fields_named) + " for this call"
        if fields_named
        else f"the operation {operation!r}"
    )
    return ProtocolError(
        code=HOST_OUTDATED,
        message=f"{host} runs an older Potpie that does not support {missing}",
        details={"host": host, "operation": operation, "fields": fields_named},
        recommended_next_action=(
            "update that host to this client's Potpie version; until then, "
            "everything it already supports keeps working"
        ),
    )


def without_fields(request: RequestT, refused: Iterable[str]) -> RequestT:
    """A copy of ``request`` with each refused (dotted) field at its default.

    For retrying a call without an optional newer field when ``host_outdated``
    names only that field. Raises ``ValueError`` for a field the request does
    not declare or one with no default, which cannot be left off.
    """

    paths: dict[str, list[str]] = {}
    for path in refused:
        head, _, rest = str(path).partition(".")
        paths.setdefault(head, [])
        if rest:
            paths[head].append(rest)
    if not is_dataclass(request) or isinstance(request, type):
        raise TypeError("without_fields needs a dataclass request")
    declared = {item.name: item for item in fields(request)}
    changes: dict[str, object] = {}
    for name, nested in paths.items():
        item = declared.get(name)
        if item is None:
            raise ValueError(f"{type(request).__name__} has no field {name!r}")
        value = getattr(request, name)
        if nested:
            changes[name] = _without_nested(value, nested)
        elif item.default is not MISSING:
            changes[name] = item.default
        elif item.default_factory is not MISSING:
            changes[name] = item.default_factory()
        else:
            raise ValueError(f"{type(request).__name__}.{name} has no default")
    return replace(request, **changes)  # type: ignore[type-var]


def _without_nested(value: object, paths: list[str]) -> object:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_without_nested(item, paths) for item in value)
    if isinstance(value, list):
        return [_without_nested(item, paths) for item in value]
    return without_fields(value, paths)


def _validate_remote_handshake(result: HandshakeResult) -> ProtocolError | None:
    """Protocol range and readiness only; the catalog fingerprint is information."""

    if not result.ready:
        return ProtocolError(
            code="host_not_ready",
            message="the managed host did not report ready state",
            details={"lifecycle_state": result.lifecycle_state},
            recommended_next_action="retry shortly",
            retry_posture="safe",
        )
    if not result.protocol_min <= PROTOCOL_VERSION <= result.protocol_max:
        return ProtocolError(
            code="protocol_version_incompatible",
            message="the managed host and this client speak different protocols",
            details={
                "client_version": PROTOCOL_VERSION,
                "host_min": result.protocol_min,
                "host_max": result.protocol_max,
            },
            recommended_next_action="install compatible Potpie versions",
        )
    return None


class _RemoteCaller:
    """Request ids, sending and the outdated-host mapping, shared by clients."""

    def __init__(
        self,
        transport: DaemonTransport,
        *,
        host: str,
        request_id_factory: RequestIdFactory | None,
    ) -> None:
        self._transport = transport
        self._host = host
        self._request_id = request_id_factory or (lambda: str(uuid4()))

    @property
    def host(self) -> str:
        return self._host

    async def send(
        self, request: ProtocolRequest, *, safety: SafetyClass | None
    ) -> ClientOutcome:
        sent = await _send_protocol_request(
            transport=self._transport, request=request, safety=safety
        )
        if isinstance(sent, Failure):
            return sent
        outcome = sent.value.outcome
        if isinstance(outcome, Failure):
            return Failure(
                host_outdated_error(
                    outcome.error, host=self._host, operation=str(request.operation)
                )
            )
        return outcome

    async def handshake(
        self,
    ) -> Success[HandshakeResult] | Failure[RuntimeBoundaryError]:
        request = HandshakeRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=self._request_id(),
            payload=HandshakePayload(
                client_protocol_min=PROTOCOL_MIN_VERSION,
                client_protocol_max=PROTOCOL_MAX_VERSION,
            ),
        )
        outcome = await self.send(request, safety=None)
        if isinstance(outcome, Failure):
            return outcome
        result = outcome.value
        if not isinstance(result, HandshakeResult):
            return Failure(
                ProtocolError(
                    code="handshake_result_malformed",
                    message="the managed host returned an invalid handshake",
                )
            )
        problem = _validate_remote_handshake(result)
        return Failure(problem) if problem is not None else Success(result)

    async def status(
        self,
    ) -> Success[DaemonStatusResult] | Failure[RuntimeBoundaryError]:
        outcome = await self.send(
            DaemonStatusRequest(
                protocol_version=PROTOCOL_VERSION,
                request_id=self._request_id(),
                payload=DaemonStatusPayload(),
            ),
            safety=None,
        )
        if isinstance(outcome, Failure):
            return outcome
        if not isinstance(outcome.value, DaemonStatusResult):
            return Failure(
                ProtocolError(
                    code="daemon_status_result_malformed",
                    message="the managed host returned an invalid status",
                )
            )
        return Success(outcome.value)

    def next_request_id(self) -> str:
        return self._request_id()


# --- engine operations ----------------------------------------------------------


class RemoteEngineClient(EngineClient):
    """The engine catalog against a managed host, for one context selector.

    No handshake is needed first; :meth:`handshake` only checks that the host
    speaks this protocol and is ready.
    """

    def __init__(
        self,
        *,
        selector: ContextSelector,
        transport: DaemonTransport,
        host: str,
        request_id_factory: RequestIdFactory | None = None,
    ) -> None:
        self._selector = selector
        self._caller = _RemoteCaller(
            transport, host=host, request_id_factory=request_id_factory
        )

    @property
    def selector(self) -> ContextSelector:
        return self._selector

    async def handshake(
        self,
    ) -> Success[HandshakeResult] | Failure[RuntimeBoundaryError]:
        return await self._caller.handshake()

    async def _dispatch(
        self,
        operation: EngineOperation,
        request: EngineRequest,
        confirmation: DestructiveConfirmation | None = None,
    ) -> ClientOutcome:
        envelope = _build_operation_request(
            selector=self._selector,
            operation=operation,
            payload=request,
            confirmation=confirmation,
            request_id=self._caller.next_request_id(),
            compatibility_ticket=None,
        )
        if isinstance(envelope, Failure):
            return envelope
        from potpie.runtime.operations import ENGINE_OPERATION_CATALOG

        return await self._caller.send(
            envelope.value, safety=ENGINE_OPERATION_CATALOG[operation].safety
        )


_AGENT_DOOR_REQUESTS: dict[type, type[EngineRequest]] = {
    AgentResolveRequest: ResolveRequest,
    AgentSearchRequest: SearchRequest,
    AgentRecordRequest: RecordRequest,
}


@overload
def from_agent_request(
    request: AgentResolveRequest,
) -> tuple[ContextSelector, ResolveRequest]: ...


@overload
def from_agent_request(
    request: AgentSearchRequest,
) -> tuple[ContextSelector, SearchRequest]: ...


@overload
def from_agent_request(
    request: AgentRecordRequest,
) -> tuple[ContextSelector, RecordRequest]: ...


def from_agent_request(
    request: AgentResolveRequest | AgentSearchRequest | AgentRecordRequest,
) -> tuple[ContextSelector, EngineRequest]:
    """An agent-door request as the selector and engine request a host takes.

    The agent door (``potpie_context_engine.core.ports.agent_context``) names
    its pot in the request; the engine catalog names it in the selector. The
    door's ``pot_id`` becomes an explicit selector and every other field is
    carried over unchanged::

        selector, engine_request = from_agent_request(door_request)
        outcome = await connection.engine(selector).resolve(engine_request)

    Raises ``TypeError`` for a request other than resolve, search or record,
    and ``ValueError`` for an empty ``pot_id``.
    """

    engine_type = _AGENT_DOOR_REQUESTS.get(type(request))
    if engine_type is None:
        raise TypeError(
            f"{type(request).__name__} is not an agent-door resolve, search or "
            "record request"
        )
    carried = {item.name: getattr(request, item.name) for item in fields(engine_type)}
    selector = ContextSelector(kind="explicit", value=request.pot_id)
    return selector, engine_type(**carried)


# --- host and extension operations ------------------------------------------------


class RemoteHostClient:
    """The ``host.*`` catalog and extension operations of one managed host.

    Every method returns a ``Success``/``Failure`` outcome, like the engine
    clients. Operations that act on a pot take a selector; the rest send the
    client's default selector, which the host ignores.
    """

    def __init__(
        self,
        target: ManagedHostEndpoint | DaemonTransport,
        *,
        selector: ContextSelector | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        request_id_factory: RequestIdFactory | None = None,
        host: str | None = None,
    ) -> None:
        if isinstance(target, ManagedHostEndpoint):
            transport: DaemonTransport = HttpsHostTransport(target, timeout_s=timeout_s)
            label = host or target.label
            self._owns_transport = True
        else:
            transport = target
            label = host or "Managed host"
            self._owns_transport = False
        self._transport = transport
        self._selector = selector or ContextSelector(kind="active")
        self._caller = _RemoteCaller(
            transport, host=label, request_id_factory=request_id_factory
        )

    @property
    def transport(self) -> DaemonTransport:
        return self._transport

    @property
    def label(self) -> str:
        """``Managed (host[:port]/prefix)``, the name used in messages."""
        return self._caller.host

    async def handshake(
        self,
    ) -> Success[HandshakeResult] | Failure[RuntimeBoundaryError]:
        return await self._caller.handshake()

    async def status(
        self,
    ) -> Success[DaemonStatusResult] | Failure[RuntimeBoundaryError]:
        return await self._caller.status()

    async def call(
        self,
        operation: HostOperation | str,
        request: object,
        *,
        selector: ContextSelector | None = None,
        confirmation: DestructiveConfirmation | None = None,
    ) -> ClientOutcome:
        """Send one host or extension operation."""

        chosen = selector or self._selector
        request_id = self._caller.next_request_id()
        name = str(operation)
        spec = (
            HOST_OPERATION_CATALOG.get(HostOperation(name))
            if name.startswith("host.")
            else None
        )
        intent = None
        if confirmation is not None:
            intent = DestructiveIntent(
                confirmed=confirmation.confirmed,
                operation=name,
                selector=chosen,
                request_id=request_id,
            )
        try:
            envelope = HostOperationRequest(
                protocol_version=PROTOCOL_VERSION,
                request_id=request_id,
                operation=name,
                selector=chosen,
                payload=request,
                destructive_intent=intent,
            )
        except ValueError as exc:
            return Failure(
                ProtocolError(
                    code="operation_payload_mismatch",
                    message=str(exc),
                    details={"operation": name},
                )
            )
        return await self._caller.send(
            envelope, safety=spec.safety if spec is not None else None
        )

    async def extension(
        self,
        name: str,
        request: object,
        result_type: type[ResultT],
        *,
        selector: ContextSelector | None = None,
    ) -> Success[ResultT] | Failure[RuntimeBoundaryError]:
        """Call an ``ext.<vendor>.<op>`` operation and decode its result."""

        outcome = await self.call(name, request, selector=selector)
        if isinstance(outcome, Failure):
            return outcome
        value = outcome.value
        plain = value.to_dict() if hasattr(value, "to_dict") else value
        try:
            return Success(result_from_payload(result_type, plain))
        except (TypeError, ValueError, KeyError):
            return Failure(
                ProtocolError(
                    code="response_result_malformed",
                    message="the extension result does not match its declared type",
                    details={"operation": name},
                )
            )

    async def pot_list(self, request: PotListRequest | None = None) -> ClientOutcome:
        return await self.call(HostOperation.POT_LIST, request or PotListRequest())

    async def pot_create(self, request: PotCreateRequest) -> ClientOutcome:
        return await self.call(HostOperation.POT_CREATE, request)

    async def pot_use(self, selector: ContextSelector) -> ClientOutcome:
        return await self.call(
            HostOperation.POT_USE, PotUseRequest(), selector=selector
        )

    async def pot_status(self, selector: ContextSelector) -> ClientOutcome:
        return await self.call(
            HostOperation.POT_STATUS, PotStatusRequest(), selector=selector
        )

    async def repo_default_set(
        self, selector: ContextSelector, request: RepoDefaultSetRequest
    ) -> ClientOutcome:
        return await self.call(
            HostOperation.REPO_DEFAULT_SET, request, selector=selector
        )

    async def repo_default_clear(
        self, request: RepoDefaultClearRequest
    ) -> ClientOutcome:
        return await self.call(HostOperation.REPO_DEFAULT_CLEAR, request)

    async def source_list(self, selector: ContextSelector) -> ClientOutcome:
        return await self.call(
            HostOperation.SOURCE_LIST, SourceListRequest(), selector=selector
        )

    async def source_add(
        self, selector: ContextSelector, request: SourceAddRequest
    ) -> ClientOutcome:
        return await self.call(HostOperation.SOURCE_ADD, request, selector=selector)

    async def whoami(self) -> ClientOutcome:
        return await self.call(HostOperation.WHOAMI, WhoamiRequest())

    async def grant_list(self, selector: ContextSelector) -> ClientOutcome:
        return await self.call(
            HostOperation.GRANT_LIST, GrantListRequest(), selector=selector
        )

    async def grant_set(
        self, selector: ContextSelector, request: GrantSetRequest
    ) -> ClientOutcome:
        return await self.call(HostOperation.GRANT_SET, request, selector=selector)

    async def grant_revoke(
        self, selector: ContextSelector, request: GrantRevokeRequest
    ) -> ClientOutcome:
        return await self.call(HostOperation.GRANT_REVOKE, request, selector=selector)

    async def aclose(self) -> None:
        if self._owns_transport:
            closer = getattr(self._transport, "aclose", None)
            if callable(closer):
                await closer()

    close = aclose


# --- one connection per host ------------------------------------------------------


class RemoteHostError(Exception):
    """A managed host refused or could not answer; ``error`` is the typed reason."""

    def __init__(self, error: RuntimeBoundaryError) -> None:
        self.error = error
        text = error.message
        if error.recommended_next_action:
            text = f"{text} ({error.recommended_next_action})"
        super().__init__(text)


class HostNotConfigured(RuntimeError):
    """``connect()`` found no managed host to connect to."""


# --- the pot directory of a managed host ------------------------------------------


Runner = Callable[[Awaitable[Any]], Any]


class _LoopThread:
    """One event loop on a daemon thread, for synchronous callers.

    A synchronous caller (a gRPC servicer thread, a script) cannot await, and
    ``asyncio.run`` per call would throw the pooled HTTPS client away each
    time. Calls are submitted to this loop instead, from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever,
                    name="potpie-managed-host",
                    daemon=True,
                )
                thread.start()
                self._loop = loop
            return self._loop

    def run(self, awaitable: Awaitable[Any], *, timeout_s: float | None = None) -> Any:
        loop = self._ensure()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise RuntimeError(
                "a synchronous managed-host call cannot run on its own event loop"
            )

        async def _await() -> Any:
            return await awaitable

        future = asyncio.run_coroutine_threadsafe(_await(), loop)
        try:
            return future.result(timeout=timeout_s)
        except BaseException:
            # A timeout or an interrupted caller: stop the call too.
            future.cancel()
            raise


_background = _LoopThread()


class ManagedPotDirectory:
    """A managed host's pots, as :class:`potpie.pots.directory.PotDirectory`.

    Every method is one or two ``host.*`` calls for the caller's actor. The
    listing (pots, the active pot and, per repo, the repo default) comes back
    from one ``host.pot_list``, so a command that reads the active pot, then
    the repo default, then the pot's name pays for one round trip, not three.

    Freshness rule: a listing, and the repo→pot index built from listings, is
    served from the cache for at most ``max_age_s`` seconds (5 by default) and
    then asked for again, so a pot another client creates, selects or shares
    shows up here within that window, however long the process lives. Every
    change made through this directory drops the cache at once. To read past
    it: ``max_age_s=0`` asks the host on every read, :meth:`invalidate` drops
    the cache, and ``listing(fresh=True)`` skips it for one read. Sources,
    grants, status and ``whoami`` are never cached.

    Threads: one directory may be shared by any number of threads. The cache
    is guarded by a lock that is never held across a host call, and a listing
    that was already in flight when a change dropped the cache is not stored.

    Failures raise :class:`RemoteHostError` carrying the host's typed error.
    Repo identities derived from a local path are never sent (they would leak
    local paths, and mean nothing on the host): reading a default for one
    answers ``None`` and binding one is refused.

    ``run`` executes one awaitable synchronously. The default runs it on a
    private background event loop, so the directory is usable from any thread;
    the CLI passes its own runner so the pooled client is shared with its
    engine calls.
    """

    origin: str = "managed"
    supports_repo_defaults: bool = True

    def __init__(
        self,
        client: RemoteHostClient,
        *,
        run: Runner | None = None,
        repo: str | None = None,
        max_age_s: float = POT_LISTING_MAX_AGE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_age_s < 0:
            raise ValueError("max_age_s must not be negative")
        self._client = client
        self._run: Runner = run or _background.run
        #: The caller's current repo identity, when it came from a git remote:
        #: the first listing asks for its default too, so the answer the next
        #: ``repo_default`` call needs is already there.
        self._repo = repo if is_remote_repo_key(repo) else None
        self._max_age_s = max_age_s
        self._clock = clock
        self._lock = threading.Lock()
        #: Bumped by every change; a read started under an older generation
        #: does not store what it fetched.
        self._generation = 0
        self._listings: dict[str | None, tuple[float, PotListResult]] = {}
        self._repo_sources: tuple[float, list[PotRepoSource]] | None = None

    @property
    def client(self) -> RemoteHostClient:
        return self._client

    @property
    def label(self) -> str:
        return self._client.label

    @property
    def max_age_s(self) -> float:
        """How long a listing is served from the cache (``0``: never)."""
        return self._max_age_s

    def invalidate(self) -> None:
        """Forget what was listed; the next read asks the host again."""
        with self._lock:
            self._generation += 1
            self._listings.clear()
            self._repo_sources = None

    def _forget_repo_sources(self) -> None:
        with self._lock:
            self._generation += 1
            self._repo_sources = None

    def _is_fresh(self, fetched_at: float) -> bool:
        return self._clock() - fetched_at < self._max_age_s

    def _cached_listing(
        self, repo: str | None, *, any_repo: bool
    ) -> PotListResult | None:
        """A young enough listing for ``repo``; the caller holds the lock."""
        entry = self._listings.get(repo)
        if entry is not None and self._is_fresh(entry[0]):
            return entry[1]
        if any_repo:
            # Pots and the active pot are the same in every listing.
            return next(
                (
                    listing
                    for fetched_at, listing in self._listings.values()
                    if self._is_fresh(fetched_at)
                ),
                None,
            )
        return None

    def close(self) -> None:
        """Close the client's pooled connections, on the loop that opened them."""
        transport = self._client.transport
        closer = getattr(transport, "aclose", None)
        if callable(closer) and getattr(transport, "is_open", True):
            self._run(closer())

    def _call(self, awaitable: Awaitable[Any]) -> Any:
        outcome = self._run(awaitable)
        if isinstance(outcome, Failure):
            raise RemoteHostError(outcome.error)
        return outcome.value

    def _listing(
        self, repo: str | None = None, *, fresh: bool = False
    ) -> PotListResult:
        asked = repo if repo is not None else self._repo
        with self._lock:
            cached = (
                None if fresh else self._cached_listing(asked, any_repo=repo is None)
            )
            if cached is not None:
                return cached
            generation = self._generation
        fetched_at = self._clock()
        listing = self._call(
            self._client.pot_list(PotListRequest(repo=asked, include_archived=True))
        )
        with self._lock:
            if generation == self._generation and self._max_age_s > 0:
                self._listings = {
                    key: entry
                    for key, entry in self._listings.items()
                    if self._is_fresh(entry[0])
                }
                self._listings[asked] = (fetched_at, listing)
        return listing

    def listing(
        self,
        *,
        repo: str | None = None,
        include_status: bool = False,
        fresh: bool = False,
    ) -> PotListResult:
        """The raw ``host.pot_list`` answer.

        ``fresh`` asks the host even when a cached listing is young enough;
        ``include_status`` always asks and is never cached.
        """
        if include_status:
            return self._call(
                self._client.pot_list(
                    PotListRequest(
                        repo=repo if is_remote_repo_key(repo) else None,
                        include_archived=True,
                        include_status=True,
                    )
                )
            )
        return self._listing(repo if is_remote_repo_key(repo) else None, fresh=fresh)

    # --- pots ---------------------------------------------------------------------
    def list_pots(self) -> list[PotInfo]:
        return list(self._listing().pots)

    def active_pot(self) -> PotInfo | None:
        listing = self._listing()
        if not listing.active_pot_id:
            return None
        for pot in listing.pots:
            if pot.pot_id == listing.active_pot_id:
                return pot if pot.active else replace(pot, active=True)
        return None

    def create_pot(
        self, *, name: str, repo: str | None = None, use: bool = False
    ) -> PotInfo:
        request = PotCreateRequest(
            name=name, repo=repo if is_remote_repo_key(repo) else None, use=use
        )
        try:
            return self._call(self._client.pot_create(request))
        finally:
            self.invalidate()

    def use_pot(self, *, ref: str) -> PotInfo:
        try:
            return self._call(
                self._client.pot_use(ContextSelector(kind="explicit", value=ref))
            )
        finally:
            self.invalidate()

    # --- repo routing defaults ----------------------------------------------------
    def repo_default(self, *, repo: str) -> str | None:
        if not is_remote_repo_key(repo):
            return None
        return self._listing(repo).repo_default_pot_id

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        _require_remote_key(repo)
        try:
            self._call(
                self._client.repo_default_set(
                    ContextSelector(kind="explicit", value=pot_id),
                    RepoDefaultSetRequest(repo=repo),
                )
            )
        finally:
            self.invalidate()

    def clear_repo_default(self, *, repo: str) -> bool:
        _require_remote_key(repo)
        try:
            result = self._call(
                self._client.repo_default_clear(RepoDefaultClearRequest(repo=repo))
            )
        finally:
            self.invalidate()
        return bool(getattr(result, "cleared", False))

    # --- sources ------------------------------------------------------------------
    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        result = self._call(
            self._client.source_list(ContextSelector(kind="explicit", value=pot_id))
        )
        return list(result.sources)

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        try:
            return self._call(
                self._client.source_add(
                    ContextSelector(kind="explicit", value=pot_id),
                    SourceAddRequest(kind=kind, location=location, name=name),
                )
            )
        finally:
            self._forget_repo_sources()

    def list_repo_sources(self) -> list[PotRepoSource]:
        """The repo→pot index, one ``host.source_list`` per live pot.

        Only routing reports need it (which pots a working tree is linked to);
        command routing itself sends a ``repository`` selector and lets the
        host match. Cached like a listing (see the class docstring).
        """
        with self._lock:
            cached = self._repo_sources
            if cached is not None and self._is_fresh(cached[0]):
                return list(cached[1])
            generation = self._generation
        fetched_at = self._clock()
        rows: list[PotRepoSource] = []
        for pot in self.list_pots():
            if pot.archived:
                continue
            rows.extend(
                PotRepoSource(
                    pot_id=pot.pot_id,
                    pot_name=pot.name,
                    name=source.name,
                    location=source.location,
                )
                for source in self.list_sources(pot_id=pot.pot_id)
                if source.kind == "repo"
            )
        with self._lock:
            if generation == self._generation and self._max_age_s > 0:
                self._repo_sources = (fetched_at, rows)
        return list(rows)

    # --- status, identity and sharing ---------------------------------------------
    def aggregate_status(self, *, pot_id: str | None = None) -> PotAggregateStatus:
        selector = (
            ContextSelector(kind="explicit", value=pot_id)
            if pot_id
            else ContextSelector(kind="active")
        )
        return self._call(self._client.pot_status(selector))

    def whoami(self) -> WhoamiResult:
        return self._call(self._client.whoami())

    def list_grants(self, *, pot_id: str) -> list[PotGrantInfo]:
        result = self._call(
            self._client.grant_list(ContextSelector(kind="explicit", value=pot_id))
        )
        return list(result.grants)

    def grant(self, *, pot_id: str, actor_id: str, role: str) -> PotGrantInfo:
        return self._call(
            self._client.grant_set(
                ContextSelector(kind="explicit", value=pot_id),
                GrantSetRequest(actor_id=actor_id, role=role),
            )
        )

    def revoke(self, *, pot_id: str, actor_id: str) -> bool:
        result = self._call(
            self._client.grant_revoke(
                ContextSelector(kind="explicit", value=pot_id),
                GrantRevokeRequest(actor_id=actor_id),
            )
        )
        return bool(getattr(result, "revoked", False))

    # --- local-only lifecycle -----------------------------------------------------
    def rename_pot(self, *, ref: str, new_name: str) -> PotInfo:
        raise _local_only("pot rename")

    def archive_pot(self, *, ref: str) -> PotInfo:
        raise _local_only("pot archive")

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo:
        raise _local_only("source status")

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        raise _local_only("source remove")

    def list_repo_defaults(self) -> dict[str, str]:
        raise _local_only("listing every repo default")


def _require_remote_key(repo: str) -> None:
    if not is_remote_repo_key(repo):
        raise ValueError(
            f"'{repo}' is a local path; a managed host binds repositories by their "
            "git remote. Run this inside a clone that has an 'origin' remote, or "
            "pass --repo <host>/<owner>/<repo>."
        )


def _local_only(command: str) -> CapabilityNotImplemented:
    return CapabilityNotImplemented(
        f"host.{command.replace(' ', '_')}",
        detail=f"'{command}' is served by the local host only, not by a managed host",
        recommended_next_action=(
            "run it against the local host with '--host local', or ask the managed "
            "host's operator"
        ),
    )


@dataclass(frozen=True, slots=True)
class HostConnection:
    """One managed host: engine clients per selector, host operations, extensions.

    Reuse it: its transport keeps one pooled, keep-alive HTTPS client per
    event loop. ``pots`` is the host's :class:`ManagedPotDirectory` for
    synchronous callers; it runs on a private background event loop, so it
    can be used from any thread while ``engine()`` clients run on the
    caller's.

    Synchronous code (an RPC servicer thread, a script) calls the async API
    through :meth:`run`, which runs it on that same background loop::

        connection = connect()
        envelope = connection.run(connection.engine(selector).resolve(request))
        answer = connection.run(connection.extension(name, request, result_type))
        me = connection.run(connection.host.whoami())
        connection.close()
    """

    endpoint: ManagedHostEndpoint
    transport: HttpsHostTransport
    host: RemoteHostClient
    pots: ManagedPotDirectory

    def engine(self, selector: ContextSelector) -> RemoteEngineClient:
        return RemoteEngineClient(
            selector=selector, transport=self.transport, host=self.endpoint.label
        )

    async def extension(
        self,
        name: str,
        request: object,
        result_type: type[ResultT],
        *,
        selector: ContextSelector | None = None,
    ) -> Success[ResultT] | Failure[RuntimeBoundaryError]:
        return await self.host.extension(name, request, result_type, selector=selector)

    async def handshake(
        self,
    ) -> Success[HandshakeResult] | Failure[RuntimeBoundaryError]:
        return await self.host.handshake()

    def run(
        self, awaitable: Awaitable[ResultT], *, timeout_s: float | None = None
    ) -> ResultT:
        """Run one of this connection's calls from synchronous code.

        The awaitable runs on the process's background managed-host event
        loop, which every thread shares (and with it the pooled connections),
        so any number of threads may call this at once; it blocks the calling
        thread until the result is in. A call still running after
        ``timeout_s`` seconds is cancelled and ``TimeoutError`` is raised.
        Async code awaits the call directly instead; calling this from the
        background loop itself raises ``RuntimeError``.
        """
        return _background.run(awaitable, timeout_s=timeout_s)

    async def aclose(self) -> None:
        await self.transport.aclose()
        await asyncio.to_thread(self.pots.close)

    def close(self) -> None:
        """:meth:`aclose` for synchronous code; later calls reconnect."""
        self.run(self.aclose())


@dataclass(frozen=True, slots=True)
class ManagedHostSettings:
    """What the host registry stores for the managed origin."""

    base_url: str
    token: str | None = field(default=None, repr=False)
    allow_insecure_host: str | None = None


_connections: dict[tuple[str, str | None, str | None, float], HostConnection] = {}


def _registered_managed_host() -> ManagedHostSettings | None:
    """The registry's usable managed host, or ``None`` (the registry hook).

    Reads ``potpie.config.host_registry.managed_host()``: its ``base_url``,
    ``token`` and ``allow_insecure_host``.
    """

    managed = host_registry.managed_host()
    if managed is None:
        return None
    return ManagedHostSettings(
        base_url=managed.base_url,
        token=managed.token or None,
        allow_insecure_host=managed.allow_insecure_host,
    )


def connect(
    origin: str | None = None,
    *,
    url: str | None = None,
    token: str | None = None,
    allow_insecure_host: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> HostConnection:
    """A connection to the managed host.

    With ``url`` the endpoint is exactly the one given. Without it, ``origin``
    must be ``None`` or ``"managed"`` and the registry's managed host is used
    (``HostNotConfigured`` when there is none). Connections are reused per
    process for the same endpoint, so their pooled client is too.
    """

    if url is None:
        if origin not in (None, "managed"):
            raise ValueError(
                f"connect() reaches a managed host; origin {origin!r} is served by "
                "the local runtime"
            )
        settings = _registered_managed_host()
        if settings is None:
            raise HostNotConfigured(
                "No managed host is configured; run 'potpie host set <url>'."
            )
        url, token, allow_insecure_host = (
            settings.base_url,
            settings.token,
            settings.allow_insecure_host,
        )
    endpoint = ManagedHostEndpoint(
        base_url=url, token=token, allow_insecure_host=allow_insecure_host
    )
    key = (endpoint.base_url, endpoint.token, endpoint.allow_insecure_host, timeout_s)
    if http_transport is None and key in _connections:
        return _connections[key]
    transport = HttpsHostTransport(
        endpoint, timeout_s=timeout_s, http_transport=http_transport
    )
    pots_transport = HttpsHostTransport(
        endpoint, timeout_s=timeout_s, http_transport=http_transport
    )
    connection = HostConnection(
        endpoint=endpoint,
        transport=transport,
        host=RemoteHostClient(transport, host=endpoint.label),
        pots=ManagedPotDirectory(RemoteHostClient(pots_transport, host=endpoint.label)),
    )
    if http_transport is None:
        _connections[key] = connection
    return connection


class ManagedHostProbe:
    """``host set --check``: handshake then ``host.pot_list``, raising on failure.

    The thin raising adapter over :class:`RemoteHostClient` for callers that
    want an exception rather than an outcome. Each method raises
    :class:`RemoteHostError` carrying the host's typed error.
    """

    def __init__(
        self,
        endpoint: ManagedHostEndpoint,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._transport = HttpsHostTransport(
            endpoint, timeout_s=timeout_s, http_transport=http_transport
        )
        self._client = RemoteHostClient(self._transport, host=endpoint.label)

    async def handshake(self) -> HandshakeResult:
        return _raise_on_failure(await self._client.handshake())

    async def pot_list(self, request: PotListRequest | None = None) -> PotListResult:
        return _raise_on_failure(await self._client.pot_list(request))

    async def aclose(self) -> None:
        await self._transport.aclose()

    close = aclose


def _raise_on_failure(outcome: Success[Any] | Failure[RuntimeBoundaryError]) -> Any:
    if isinstance(outcome, Failure):
        raise RemoteHostError(outcome.error)
    return outcome.value


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "HOST_OUTDATED",
    "HostConnection",
    "HostNotConfigured",
    "HttpsHostTransport",
    "ManagedHostEndpoint",
    "ManagedHostProbe",
    "ManagedPotDirectory",
    "ManagedHostSettings",
    "RemoteEngineClient",
    "RemoteHostClient",
    "RemoteHostError",
    "POT_LISTING_MAX_AGE_S",
    "connect",
    "from_agent_request",
    "host_outdated_error",
    "without_fields",
]
