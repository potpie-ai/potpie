"""Discovery helpers for the root HTTP/RPC daemon.

The legacy UDS operation transport still has a test-only client below, but the
active CLI daemon path uses ``base_url`` + bearer token discovery exclusively.
"""

from __future__ import annotations

import json
import pathlib

import httpx

from potpie_context_engine.domain.ports.daemon.lifecycle import DaemonDiscovery


def discovery_path(home: pathlib.Path) -> pathlib.Path:
    return home / "discovery.json"


def load_discovery(home: pathlib.Path) -> DaemonDiscovery | None:
    p = discovery_path(home)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return parse_discovery(raw)


def parse_discovery(raw: dict[object, object]) -> DaemonDiscovery | None:
    """Parse active daemon discovery JSON without flattening typed fields to strings."""
    discovery: DaemonDiscovery = {}
    transport = raw.get("transport")
    if isinstance(transport, str):
        discovery["transport"] = transport
    base_url = raw.get("base_url")
    if isinstance(base_url, str):
        discovery["base_url"] = base_url
    token = raw.get("token")
    if isinstance(token, str):
        discovery["token"] = token
    log_file = raw.get("log_file")
    if isinstance(log_file, str):
        discovery["log_file"] = log_file
    backend = raw.get("backend")
    if isinstance(backend, str):
        discovery["backend"] = backend
    pid = raw.get("pid")
    if isinstance(pid, int):
        discovery["pid"] = pid
    elif isinstance(pid, str):
        try:
            discovery["pid"] = int(pid)
        except ValueError:
            pass
    return discovery


def legacy_client_for(home: pathlib.Path) -> httpx.Client:
    """Return a client for the legacy UDS/TCP operation runtime.

    This is intentionally not used by the public CLI daemon commands. It remains
    for direct tests of the legacy operation transport while the active daemon
    contract is HTTP/RPC discovery.
    """
    return _legacy_client_for(home)


def _tcp_base_url(bind: str) -> str:
    rest = bind[len("tcp:") :]
    if rest.startswith("["):
        bracket_end = rest.find("]")
        if bracket_end == -1 or len(rest) <= bracket_end + 2:
            raise RuntimeError(f"malformed tcp bind {bind!r}")
        if rest[bracket_end + 1] != ":":
            raise RuntimeError(f"malformed tcp bind {bind!r}")
        port = rest[bracket_end + 2 :]
        return f"http://{rest[: bracket_end + 1]}:{port}"
    host, sep, port = rest.rpartition(":")
    if not sep or not host or not port:
        raise RuntimeError(f"malformed tcp bind {bind!r}")
    return f"http://{host}:{port}"


def _legacy_client_for(home: pathlib.Path) -> httpx.Client:
    path = discovery_path(home)
    if not path.exists():
        raise RuntimeError("daemon not running (no discovery file)")
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        raise RuntimeError("daemon not running (invalid discovery file)")
    if not isinstance(raw, dict):
        raise RuntimeError("daemon not running (invalid discovery file)")
    bind = raw.get("bind")
    if not isinstance(bind, str) or not bind:
        raise RuntimeError("discovery file missing 'bind' field")
    if bind.startswith("unix:"):
        sock = bind[len("unix:") :]
        return httpx.Client(
            transport=httpx.HTTPTransport(uds=sock), base_url="http://localhost"
        )
    if bind.startswith("tcp:"):
        return httpx.Client(base_url=_tcp_base_url(bind))
    raise RuntimeError(f"unknown bind {bind!r}")
