"""HTTP client for the running daemon (UDS-preferred)."""
from __future__ import annotations
import json, pathlib, httpx


def discovery_path(home: pathlib.Path) -> pathlib.Path:
    return home / "discovery.json"


def load_discovery(home: pathlib.Path) -> dict | None:
    p = discovery_path(home)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def client_for(home: pathlib.Path) -> httpx.Client:
    """Return an httpx.Client connected to the running daemon. The caller owns and must close it
    (use `with client_for(home) as c:`)."""
    d = load_discovery(home)
    if not d:
        raise RuntimeError("daemon not running (no discovery file)")
    bind = d["bind"]
    if bind.startswith("unix:"):
        sock = bind[len("unix:"):]
        return httpx.Client(transport=httpx.HTTPTransport(uds=sock), base_url="http://localhost")
    if bind.startswith("tcp:"):
        _, host, port = bind.split(":", 2)
        return httpx.Client(base_url=f"http://{host}:{port}")
    raise RuntimeError(f"unknown bind {bind!r}")
