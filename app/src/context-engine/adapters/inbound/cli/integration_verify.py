"""Lightweight read-only API probes for stored integration credentials."""

from __future__ import annotations

import time
from typing import Any

import httpx

from adapters.inbound.cli.provider_config import Provider


def verify_integration_access(
    provider: Provider,
    credentials: dict[str, Any],
) -> tuple[bool, str]:
    """Return ``(ok, message)`` after a minimal read-only API check."""
    if provider == "linear":
        access_token = str(credentials.get("access_token") or "").strip()
        if not access_token:
            return False, "not authenticated"
        expires_at = credentials.get("expires_at")
        if expires_at is not None:
            try:
                if time.time() > float(expires_at):
                    return False, "access token expired"
            except (TypeError, ValueError):
                pass
        return _verify_linear(access_token)
    return False, f"unknown provider {provider!r}"


def _verify_linear(access_token: str) -> tuple[bool, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    query = "query { viewer { id name email organization { name } } }"
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                "https://api.linear.app/graphql",
                headers=headers,
                json={"query": query},
            )
    except httpx.HTTPError:
        return False, "Linear API request failed"
    if response.status_code != 200:
        return False, f"Linear API HTTP {response.status_code}"
    try:
        data = response.json()
    except ValueError:
        return False, "Linear API returned non-JSON response"
    viewer = (data.get("data") or {}).get("viewer") if isinstance(data, dict) else None
    if not isinstance(viewer, dict):
        errors = data.get("errors") if isinstance(data, dict) else None
        if errors:
            return False, "Linear API rejected token"
        return False, "Linear viewer unavailable"
    name = viewer.get("name") or viewer.get("email") or viewer.get("id") or "user"
    org_payload = viewer.get("organization")
    org = org_payload.get("name") if isinstance(org_payload, dict) else None
    if org:
        return True, f"ok ({name} @ {org})"
    return True, f"ok ({name})"
