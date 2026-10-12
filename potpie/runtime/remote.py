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
import importlib
import ipaddress
import sys
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from typing import Any, TypeVar
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

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
    PotListRequest,
    PotListResult,
    PotStatusRequest,
    PotUseRequest,
    RepoDefaultClearRequest,
    RepoDefaultSetRequest,
    SourceAddRequest,
    SourceListRequest,
    WhoamiRequest,
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
from potpie_context_engine.requests import EngineRequest
from potpie_context_engine.results import result_from_payload

HOST_OUTDATED = "host_outdated"
DEFAULT_TIMEOUT_S = 30.0
_OPERATIONS_PATH = "v1/operations"
_OUTDATED_SOURCES = frozenset({"operation_unknown", "request_fields_unsupported"})

ResultT = TypeVar("ResultT")
RequestT = TypeVar("RequestT")


# --- addresses ---------------------------------------------------------------------


def _is_loopback(hostname: str | None) -> bool:
    host = (hostname or "").strip().lower().strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_REGISTRY_MODULE = "potpie.config.host_registry"


def _host_registry() -> Any | None:
    """The host registry module, or ``None`` in a build without it."""

    try:
        return importlib.import_module(_REGISTRY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _REGISTRY_MODULE:
            raise
        return None


def _normalize_managed_url(url: str, allow_insecure_host: str | None) -> str:
    """The registry's URL rules, the single definition of what is acceptable."""

    host_registry = _host_registry()
    if host_registry is None:
        return _fallback_normalize_managed_url(url, allow_insecure_host)
    return host_registry.normalize_managed_url(
        url, allow_insecure_host=allow_insecure_host
    )


def _fallback_normalize_managed_url(url: str, allow_insecure_host: str | None) -> str:
    # TEMPORARY: used only until potpie.config.host_registry is on this branch;
    # delete it then, so the registry owns the one copy of these rules.
    raw = (url or "").strip()
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(
            f"Managed host URL {raw!r} needs an http(s) scheme and a host."
        )
    parts.port  # noqa: B018 - raises ValueError for an invalid port
    if parts.scheme == "http" and not _is_loopback(parts.hostname):
        host = parts.hostname.lower()
        key = f"[{host}]" if ":" in host else host
        key = f"{key}:{parts.port}" if parts.port is not None else key
        if (allow_insecure_host or "").strip().lower() != key:
            raise ValueError(
                f"Managed host URL {raw!r} uses plain http:// for a non-loopback "
                "host; use https://, or opt in exactly this host[:port]."
            )
    return raw.rstrip("/")


def _warn_insecure_transport(base_url: str) -> None:
    host_registry = _host_registry()
    if host_registry is None:
        _fallback_warn_insecure_transport(base_url)
        return
    host_registry.warn_insecure_transport(base_url)


_warned_insecure: set[str] = set()


def _fallback_warn_insecure_transport(base_url: str) -> None:
    # TEMPORARY: see _fallback_normalize_managed_url.
    parts = urlsplit(base_url)
    if parts.scheme != "http" or _is_loopback(parts.hostname):
        return
    if base_url in _warned_insecure:
        return
    _warned_insecure.add(base_url)
    print(
        f"warning: {base_url} is reached over plain http (insecure opt-in); "
        "its API key is sent unencrypted",
        file=sys.stderr,
        flush=True,
    )


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
        if token is None and not _is_loopback(urlsplit(normalized).hostname):
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
        return parts.scheme == "http" and not _is_loopback(parts.hostname)

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
    """Send typed requests to one managed host over one pooled HTTPS client.

    Satisfies ``DaemonTransport.send``. The ``httpx.AsyncClient`` keeps its
    connections alive across calls and is rebuilt only when it is used from a
    different event loop. Transport failures raise ``TransportFailure`` with
    ``dispatched`` saying whether the host may have acted.
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
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    async def send(self, request: ProtocolRequest) -> ProtocolResponse:
        client = self._get_client()
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

    async def aclose(self) -> None:
        client, self._client = self._client, None
        self._client_loop = None
        if client is not None:
            await client.aclose()

    close = aclose

    def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._http_transport,
                follow_redirects=False,
            )
            self._client_loop = loop
        return self._client


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


@dataclass(frozen=True, slots=True)
class HostConnection:
    """One managed host: engine clients per selector, host operations, extensions.

    Reuse it: its transport keeps one pooled, keep-alive HTTPS client.
    """

    endpoint: ManagedHostEndpoint
    transport: HttpsHostTransport
    host: RemoteHostClient

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

    async def aclose(self) -> None:
        await self.transport.aclose()


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
    ``token`` and ``allow_insecure_host``. Without the registry module there is
    no configured host.
    """

    host_registry = _host_registry()
    if host_registry is None:
        return None
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
    connection = HostConnection(
        endpoint=endpoint,
        transport=transport,
        host=RemoteHostClient(transport, host=endpoint.label),
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
    "ManagedHostSettings",
    "RemoteEngineClient",
    "RemoteHostClient",
    "RemoteHostError",
    "connect",
    "host_outdated_error",
    "without_fields",
]
