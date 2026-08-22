"""Authenticated local HTTP transport for the typed daemon protocol."""

from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from potpie.runtime.codec import decode_response, encode_request
from potpie.runtime.protocol import (
    FailureResponse,
    ProtocolRequest,
    ProtocolResponse,
    TransportFailure,
)
from potpie_context_engine import Failure


@dataclass(frozen=True, slots=True)
class RuntimeEndpoint:
    kind: Literal["uds", "tcp"]
    address: str
    port: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "uds":
            if self.port is not None:
                raise ValueError("Unix-domain endpoints cannot carry a TCP port")
            if not Path(self.address).is_absolute():
                raise ValueError("Unix-domain socket paths must be absolute")
            return
        try:
            address = ipaddress.ip_address(self.address)
        except ValueError as exc:
            raise ValueError("TCP endpoint address must be a loopback IP") from exc
        if not address.is_loopback:
            raise ValueError("daemon TCP endpoint must be loopback-only")
        if self.port is None or not 0 < self.port <= 65535:
            raise ValueError("TCP endpoint requires a valid port")

    @property
    def display(self) -> str:
        if self.kind == "uds":
            return self.address
        host = f"[{self.address}]" if ":" in self.address else self.address
        return f"{host}:{self.port}"


class HttpDaemonTransport:
    """Send one typed protocol over authenticated UDS or loopback TCP HTTP."""

    def __init__(
        self,
        *,
        endpoint: RuntimeEndpoint,
        bearer_token: str,
        timeout_s: float = 30.0,
    ) -> None:
        if not bearer_token:
            raise ValueError("daemon bearer token must not be empty")
        if timeout_s <= 0:
            raise ValueError("daemon timeout must be positive")
        self.endpoint = endpoint
        self._bearer_token = bearer_token
        self._timeout = httpx.Timeout(timeout_s)
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def send(self, request: ProtocolRequest) -> ProtocolResponse:
        client = await self._get_client()
        try:
            response = await client.post(
                "/v1/operations",
                json=encode_request(request),
                headers={"Authorization": f"Bearer {self._bearer_token}"},
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise TransportFailure(code="daemon_unavailable", dispatched=False) from exc
        except (
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.WriteError,
            httpx.WriteTimeout,
        ) as exc:
            raise TransportFailure(
                code="daemon_connection_lost", dispatched=True
            ) from exc
        except httpx.RequestError as exc:
            raise TransportFailure(
                code="daemon_transport_failed", dispatched=True
            ) from exc

        try:
            document = response.json()
        except ValueError as exc:
            raise TransportFailure(
                code="daemon_response_not_json", dispatched=True
            ) from exc
        decoded = decode_response(document, request=request)
        if isinstance(decoded, Failure):
            return FailureResponse(
                protocol_version=request.protocol_version,
                request_id=request.request_id,
                outcome=decoded,
            )
        return decoded.value

    async def close(self) -> None:
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> HttpDaemonTransport:
        await self._get_client()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is not None:
                return self._client
            if self.endpoint.kind == "uds":
                transport = httpx.AsyncHTTPTransport(uds=self.endpoint.address)
                base_url = "http://localhost"
            else:
                transport = httpx.AsyncHTTPTransport()
                base_url = f"http://{self.endpoint.display}"
            self._client = httpx.AsyncClient(
                transport=transport,
                base_url=base_url,
                timeout=self._timeout,
            )
            return self._client


__all__ = ["HttpDaemonTransport", "RuntimeEndpoint"]
