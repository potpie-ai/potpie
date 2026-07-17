"""Daemon-backed ``HostShell`` client for the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.domain.errors import CapabilityNotImplemented, ContextEngineDisabled, PotNotFound
from potpie.daemon.daemon import Daemon
from potpie.daemon.daemon_rpc import decode, encode


@dataclass(slots=True)
class DaemonRpcClient:
    """Small local HTTP client that calls operations inside the daemon."""

    daemon: Daemon = field(
        default_factory=lambda: Daemon(home=default_home(), in_process=False)
    )
    timeout_s: float = 30.0

    def call(self, surface: str, method: str, *args: Any, **kwargs: Any) -> Any:
        discovery = self._rpc_discovery()
        url = f"{discovery['base_url'].rstrip('/')}/rpc"
        payload = {
            "surface": surface,
            "method": method,
            "args": encode(args),
            "kwargs": encode(kwargs),
        }
        try:
            response = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {discovery['token']}"},
                timeout=self.timeout_s,
            )
        except httpx.RequestError as exc:
            raise ContextEngineDisabled(f"Potpie daemon is unavailable: {exc}") from exc
        data = _response_json(response)
        if response.status_code >= 400 or not data.get("ok", False):
            _raise_remote_error(data)
        return decode(data.get("result"))

    def attr(self, surface: str, name: str) -> Any:
        discovery = self._rpc_discovery()
        url = f"{discovery['base_url'].rstrip('/')}/attr"
        try:
            response = httpx.post(
                url,
                json={"surface": surface, "name": name},
                headers={"Authorization": f"Bearer {discovery['token']}"},
                timeout=self.timeout_s,
            )
        except httpx.RequestError as exc:
            raise ContextEngineDisabled(f"Potpie daemon is unavailable: {exc}") from exc
        data = _response_json(response)
        if response.status_code >= 400 or not data.get("ok", False):
            _raise_remote_error(data)
        return decode(data.get("result"))

    def _rpc_discovery(self) -> dict[str, str]:
        discovery = self.daemon.discovery()
        if discovery is None:
            raise ContextEngineDisabled(
                "Potpie daemon is not running. Run 'potpie setup' to start it."
            )
        if not discovery.get("base_url") or not discovery.get("token"):
            raise ContextEngineDisabled(
                "Potpie daemon is running but does not expose the CLI RPC surface. "
                "Run 'potpie daemon restart'."
            )
        return discovery


class RemoteSurface:
    """Dynamic proxy for a ``HostShell`` surface or nested capability port."""

    _NESTED = frozenset(
        {
            "mutation",
            "claim_query",
            "semantic",
            "inspection",
            "analytics",
            "snapshot",
        }
    )
    _REMOTE_ATTRS = frozenset({"profile"})

    def __init__(self, client: DaemonRpcClient, path: str) -> None:
        self._client = client
        self._path = path

    def __getattr__(self, name: str) -> Any:
        if name in self._NESTED:
            return RemoteSurface(self._client, f"{self._path}.{name}")
        if name in self._REMOTE_ATTRS:
            return self._client.attr(self._path, name)

        def _call(*args: Any, **kwargs: Any) -> Any:
            return self._client.call(self._path, name, *args, **kwargs)

        return _call


@dataclass
class RemoteHostShell:
    """CLI facade whose service calls are executed inside the daemon."""

    rpc: DaemonRpcClient = field(default_factory=DaemonRpcClient)
    profile: str = "local"

    def __post_init__(self) -> None:
        self.daemon = self.rpc.daemon
        self.agent_context = RemoteSurface(self.rpc, "agent_context")
        self.graph = RemoteSurface(self.rpc, "graph")
        self.graph_workbench = RemoteSurface(self.rpc, "graph_workbench")
        self.pots = RemoteSurface(self.rpc, "pots")
        self.skills = RemoteSurface(self.rpc, "skills")
        self.backend = RemoteSurface(self.rpc, "backend")
        self.ledger = RemoteSurface(self.rpc, "ledger")
        self.nudge = RemoteSurface(self.rpc, "nudge")
        self.config = RemoteSurface(self.rpc, "config")
        self.installer = RemoteSurface(self.rpc, "installer")
        self.auth = RemoteSurface(self.rpc, "auth")
        self.setup = RemoteSurface(self.rpc, "setup")


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ContextEngineDisabled(
            f"Potpie daemon returned a non-JSON response ({response.status_code})."
        ) from exc
    if not isinstance(data, dict):
        raise ContextEngineDisabled("Potpie daemon returned an invalid response.")
    return data


def _raise_remote_error(data: dict[str, Any]) -> None:
    error = data.get("error") or {}
    code = str(error.get("code") or "daemon_error")
    message = str(error.get("message") or "Potpie daemon request failed.")
    detail = error.get("detail")
    next_action = error.get("recommended_next_action")
    if code == "not_implemented":
        raise CapabilityNotImplemented(
            str(error.get("capability") or message),
            detail=detail,
            recommended_next_action=next_action,
        )
    if code == "pot_not_found":
        raise PotNotFound(message)
    if code in {"validation_error", "value_error"}:
        exc = ValueError(message)
        # Re-attach structured guidance so the CLI error boundary can surface
        # detail/recommended_next_action exactly as with an in-process service.
        if detail is not None:
            setattr(exc, "detail", detail)
        if next_action is not None:
            setattr(exc, "recommended_next_action", next_action)
        raise exc
    raise ContextEngineDisabled(message)


__all__ = ["DaemonRpcClient", "RemoteHostShell", "RemoteSurface"]
