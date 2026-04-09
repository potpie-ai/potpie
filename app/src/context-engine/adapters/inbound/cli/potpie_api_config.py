"""Resolve Potpie API base URL and API key for thin CLI / MCP (X-API-Key on /api/v2)."""

from __future__ import annotations

import os

from adapters.inbound.cli.credentials_store import get_stored_api_base_url, get_stored_api_key


def resolve_potpie_api_base_url() -> str:
    """Base URL only (no path), no trailing slash."""
    u = (
        os.getenv("POTPIE_API_URL")
        or os.getenv("POTPIE_BASE_URL")
        or get_stored_api_base_url()
        or ""
    ).strip()
    port = (os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT") or "").strip()
    if not u and port:
        u = f"http://127.0.0.1:{port}"
    u = u.rstrip("/")
    if not u:
        raise ValueError(
            "Potpie API base URL missing. Set POTPIE_API_URL or POTPIE_BASE_URL, "
            "use `context-engine login --url …`, or set POTPIE_PORT for http://127.0.0.1:<port>."
        )
    return u


def resolve_potpie_api_key() -> str:
    k = (os.getenv("POTPIE_API_KEY") or get_stored_api_key() or "").strip()
    if not k:
        raise ValueError(
            "Potpie API key missing. Set POTPIE_API_KEY or run `context-engine login <key>`."
        )
    return k
