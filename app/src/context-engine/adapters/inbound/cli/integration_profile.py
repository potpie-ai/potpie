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
    try:
        data = response.json()
    except ValueError:
        return {}
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


def _profile_section(
    profile: dict[str, Any],
    prior: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Prefer a profile subsection, then fall back to prior stored metadata."""
    section = profile.get(key)
    if isinstance(section, dict) and section:
        return dict(section)
    prior_section = prior.get(key)
    if isinstance(prior_section, dict):
        return dict(prior_section)
    return {}


def _resolve_stored_at(tokens: dict[str, Any], prior: dict[str, Any]) -> Any:
    stored_at = tokens.get("stored_at")
    if stored_at is None:
        stored_at = prior.get("stored_at")
    if stored_at is None:
        stored_at = time.time()
    return stored_at


def _apply_optional_fields(
    record: dict[str, Any],
    tokens: dict[str, Any],
    prior: dict[str, Any],
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        value = tokens.get(field)
        if value is None:
            value = prior.get(field)
        if value is not None:
            record[field] = value


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
    account = _profile_section(profile, prior, "account")
    organization = _profile_section(profile, prior, "organization")
    scopes = _normalize_scopes(tokens.get("scope") or prior.get("scopes") or prior.get("scope"))
    stored_at = _resolve_stored_at(tokens, prior)

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
    _apply_optional_fields(record, tokens, prior, ("expires_at", "expires_in", "cloud_id"))
    return record




def linear_account_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    account = entry.get("account")
    if isinstance(account, dict):
        return account
    return {}
