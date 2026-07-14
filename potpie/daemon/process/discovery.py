"""Read and validate discovery metadata for the active HTTP/RPC daemon."""

from __future__ import annotations

import json
from pathlib import Path

from potpie.daemon.contracts import DaemonDiscovery


def discovery_path(home: Path) -> Path:
    return home / "discovery.json"


def load_discovery(home: Path) -> DaemonDiscovery | None:
    path = discovery_path(home)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return parse_discovery(raw)


def parse_discovery(raw: dict[object, object]) -> DaemonDiscovery:
    """Keep only typed fields understood by the active daemon contract."""

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


__all__ = ["discovery_path", "load_discovery", "parse_discovery"]
