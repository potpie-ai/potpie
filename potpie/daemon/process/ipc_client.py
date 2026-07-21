"""HTTP client for the running daemon (UDS-preferred)."""

from __future__ import annotations

import json
import pathlib

import httpx


def discovery_path(home: pathlib.Path) -> pathlib.Path:
    return home / "discovery.json"


def load_discovery(home: pathlib.Path) -> dict[str, str] | None:
    p = discovery_path(home)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return {str(key): str(value) for key, value in raw.items()}


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


def client_for(home: pathlib.Path) -> httpx.Client:
    """Return an httpx.Client connected to the running daemon. The caller owns and must close it
    (use `with client_for(home) as c:`)."""
    d = load_discovery(home)
    if not d:
        raise RuntimeError("daemon not running (no discovery file)")
    bind = d.get("bind")
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
