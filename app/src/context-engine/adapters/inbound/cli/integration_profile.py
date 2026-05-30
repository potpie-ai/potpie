"""Fetch and normalize integration identity metadata for credentials.json."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

_LINEAR_VIEWER_QUERY = "query { viewer { id name email organization { id name } } }"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_linear_viewer(access_token: str) -> dict[str, Any]:
    """Return ``account`` and ``organization`` dicts from Linear's GraphQL API."""
    token = access_token.strip()
    if not token:
        return {}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                "https://api.linear.app/graphql",
                headers=headers,
                json={"query": _LINEAR_VIEWER_QUERY},
            )
    except httpx.HTTPError:
        return {}
    if response.status_code != 200:
        return {}
    data = response.json()
    viewer = (data.get("data") or {}).get("viewer") if isinstance(data, dict) else None
    if not isinstance(viewer, dict):
        return {}

    account: dict[str, Any] = {}
    viewer_id = viewer.get("id")
    if viewer_id is not None:
        account["id"] = str(viewer_id)
    name = viewer.get("name")
    if name:
        account["name"] = str(name)
    email = viewer.get("email")
    if email:
        account["email"] = str(email)

    organization: dict[str, Any] = {}
    org_payload = viewer.get("organization")
    if isinstance(org_payload, dict):
        org_id = org_payload.get("id")
        if org_id is not None:
            organization["id"] = str(org_id)
        org_name = org_payload.get("name")
        if org_name:
            organization["name"] = str(org_name)

    out: dict[str, Any] = {}
    if account:
        out["account"] = account
    if organization:
        out["organization"] = organization
    return out


def _normalize_scopes(scope: Any) -> list[str]:
    if isinstance(scope, list):
        return [str(s).strip() for s in scope if str(s).strip()]
    if isinstance(scope, str) and scope.strip():
        return [part.strip() for part in scope.split(",") if part.strip()]
    return []


def build_linear_integration_record(
    tokens: dict[str, Any],
    *,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build non-secret Linear metadata aligned with GitHub's credentials shape."""
    prior = dict(existing or {})
    now = utc_now_iso()
    access_token = str(tokens.get("access_token") or "").strip()

    profile = fetch_linear_viewer(access_token) if access_token else {}
    account = profile.get("account") if isinstance(profile.get("account"), dict) else {}
    if not account and isinstance(prior.get("account"), dict):
        account = dict(prior["account"])

    organization = (
        profile.get("organization")
        if isinstance(profile.get("organization"), dict)
        else {}
    )
    if not organization and isinstance(prior.get("organization"), dict):
        organization = dict(prior["organization"])

    scopes = _normalize_scopes(tokens.get("scope") or prior.get("scopes") or prior.get("scope"))
    stored_at = tokens.get("stored_at")
    if stored_at is None:
        stored_at = prior.get("stored_at")
    if stored_at is None:
        stored_at = time.time()

    record: dict[str, Any] = {
        "provider": "linear",
        "provider_host": "linear.app",
        "auth_type": "oauth",
        "token_storage": "keychain",
        "token_type": tokens.get("token_type") or prior.get("token_type") or "Bearer",
        "stored_at": stored_at,
        "created_at": prior.get("created_at") or now,
        "updated_at": now,
        "metadata": {"auth_flow": "pkce"},
    }
    if account:
        record["account"] = account
    if organization:
        record["organization"] = organization
    if scopes:
        record["scopes"] = scopes
    for field in ("expires_at", "expires_in", "cloud_id"):
        value = tokens.get(field)
        if value is None:
            value = prior.get(field)
        if value is not None:
            record[field] = value
    return record


def build_product_integration_record(
    product: str, credentials: dict[str, Any]
) -> dict[str, Any]:
    """Build metadata for a single Atlassian product (jira or confluence)."""
    record = build_atlassian_integration_record(credentials)
    record["provider"] = product.strip().lower()
    return record


def build_atlassian_integration_record(credentials: dict[str, Any]) -> dict[str, Any]:
    """Build non-secret Atlassian metadata with nested account/site plus flat fields."""
    now = utc_now_iso()
    email = str(credentials.get("email") or "").strip()
    cloud_id = str(credentials.get("cloud_id") or "").strip()
    site_url = str(credentials.get("site_url") or "").strip()
    site_name = str(credentials.get("site_name") or "").strip()

    account: dict[str, Any] = {}
    if email:
        account["email"] = email

    site: dict[str, Any] = {}
    if cloud_id:
        site["cloud_id"] = cloud_id
    if site_url:
        site["site_url"] = site_url
    if site_name:
        site["site_name"] = site_name

    stored_at = credentials.get("stored_at")
    if stored_at is None:
        stored_at = time.time()

    token_style = str(credentials.get("token_style") or "classic").strip() or "classic"
    record: dict[str, Any] = {
        "provider": "atlassian",
        "provider_host": "atlassian.net",
        "auth_type": "api_token",
        "token_style": token_style,
        "token_storage": "keychain",
        "stored_at": stored_at,
        "created_at": credentials.get("created_at") or now,
        "updated_at": now,
        "metadata": {"auth_flow": "classic_api_token"},
    }
    if account:
        record["account"] = account
    if site:
        record["site"] = site
    if email:
        record["email"] = email
    if cloud_id:
        record["cloud_id"] = cloud_id
    if site_url:
        record["site_url"] = site_url
    if site_name:
        record["site_name"] = site_name
    workspaces = credentials.get("workspaces")
    if isinstance(workspaces, dict) and workspaces:
        record["workspaces"] = workspaces
    return record


def atlassian_workspaces_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    workspaces = entry.get("workspaces")
    if isinstance(workspaces, dict):
        return workspaces
    return {}


def linear_account_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    account = entry.get("account")
    if isinstance(account, dict):
        return account
    return {}


def atlassian_account_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    account = entry.get("account")
    if isinstance(account, dict):
        return account
    email = entry.get("email")
    if email:
        return {"email": email}
    return {}


def atlassian_site_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    site = entry.get("site")
    if isinstance(site, dict):
        return site
    out: dict[str, Any] = {}
    for key in ("cloud_id", "site_url", "site_name"):
        value = entry.get(key)
        if value:
            out[key] = value
    return out
