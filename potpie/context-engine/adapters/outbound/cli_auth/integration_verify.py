"""Lightweight read-only API probes for stored integration credentials."""

from __future__ import annotations

import time
from typing import Any

from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.atlassian_client import (
    AtlassianAuthErrorKind,
    fetch_cloud_id_for_site,
    normalize_site_url,
    verify_gateway_product,
)
from adapters.outbound.cli_auth.provider_config import Provider


def verify_integration_access(
    provider: Provider,
    credentials: dict[str, Any],
    *,
    http: HttpClient | None = None,
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
        return _verify_linear(access_token, http=http)
    if provider == "github":
        return _verify_github(credentials, http=http)
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


def _verify_linear(
    access_token: str, *, http: HttpClient | None = None
) -> tuple[bool, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    query = "query { viewer { id name email organization { name } } }"
    owns = http is None
    http = http or AuthHttpClient(timeout=15.0)
    try:
        response = http.post(
            "https://api.linear.app/graphql",
            headers=headers,
            json={"query": query},
        )
    except AuthHttpError:
        return False, "Linear API request failed"
    finally:
        if owns:
            http.close()
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


def _verify_github(
    credentials: dict[str, Any],
    *,
    http: HttpClient | None = None,
) -> tuple[bool, str]:
    from adapters.outbound.cli_auth.github import GitHubDeviceFlowError, verify_account

    access_token = str(credentials.get("access_token") or "").strip()
    if not access_token:
        return False, "not authenticated"
    try:
        account = verify_account(access_token, http=http)
    except GitHubDeviceFlowError as exc:
        return False, str(exc)
    login = account.login
    if account.email:
        return True, f"ok ({login} <{account.email}>)"
    return True, f"ok ({login})"


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
