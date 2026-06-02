"""Lightweight read-only API probes for stored integration credentials."""

from __future__ import annotations

import time
from typing import Any

import httpx

from adapters.inbound.cli.atlassian_auth import (
    AtlassianAuthErrorKind,
    fetch_cloud_id_for_site,
    normalize_site_url,
    verify_gateway_product,
)
from adapters.inbound.cli.provider_config import Provider


def verify_integration_access(
    provider: Provider,
    credentials: dict[str, Any],
) -> tuple[bool, str]:
    """Return ``(ok, message)`` after a minimal read-only API check."""
    if provider == "atlassian":
        jira_ok, jira_message = _verify_atlassian_product("jira", credentials)
        if jira_ok:
            return jira_ok, jira_message
        confluence_ok, confluence_message = _verify_atlassian_product(
            "confluence",
            credentials,
        )
        if confluence_ok:
            return confluence_ok, confluence_message
        return False, f"jira: {jira_message}; confluence: {confluence_message}"
    if provider in ("jira", "confluence"):
        return _verify_atlassian_product(provider, credentials)
    return False, f"unknown provider {provider!r}"


def _verify_message_for_kind(
    provider: Provider,
    kind: AtlassianAuthErrorKind | None,
) -> str:
    if kind == AtlassianAuthErrorKind.INVALID_CREDENTIALS:
        return "invalid Atlassian email or API token"
    if kind == AtlassianAuthErrorKind.INSUFFICIENT_SCOPES:
        return f"{provider} API token missing required read scopes"
    if kind == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED:
        return "could not resolve Atlassian cloud ID for site"
    if kind == AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED:
        return f"no {provider} access on this site (license or scope)"
    return f"{provider} gateway verification failed"


def _verify_atlassian_product(
    provider: Provider,
    credentials: dict[str, Any],
) -> tuple[bool, str]:
    email = str(credentials.get("email") or "").strip()
    api_token = str(credentials.get("api_token") or "").strip()
    site_url = normalize_site_url(str(credentials.get("site_url") or ""))
    if not email or not api_token or not site_url:
        return False, "not authenticated"

    cloud_id = str(credentials.get("cloud_id") or "").strip()
    if not cloud_id:
        cloud_id = fetch_cloud_id_for_site(site_url)
    if not cloud_id:
        return False, "could not resolve Atlassian cloud ID for site"

    result = verify_gateway_product(email, api_token, cloud_id, provider)
    if not result.ok:
        return False, _verify_message_for_kind(provider, result.error_kind)

    site_name = credentials.get("site_name") or site_url
    name = result.display_name or "user"
    return True, f"ok ({name} @ {site_name})"


