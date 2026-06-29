"""Fetch and normalize integration identity metadata for credentials.json."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient

_LINEAR_VIEWER_QUERY = (
    "query { viewer { id name email organization { id name urlKey } } }"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_linear_viewer(
    access_token: str, *, http: HttpClient | None = None
) -> dict[str, Any]:
    """Return ``account`` and ``organization`` dicts from Linear's GraphQL API."""
    token = access_token.strip()
    if not token:
        return {}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    owns = http is None
    http = http or AuthHttpClient(timeout=15.0)
    try:
        response = http.post(
            "https://api.linear.app/graphql",
            headers=headers,
            json={"query": _LINEAR_VIEWER_QUERY},
        )
    except AuthHttpError:
        return {}
    finally:
        if owns:
            http.close()
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
        url_key = org_payload.get("urlKey") or org_payload.get("url_key")
        if url_key:
            organization["url_key"] = str(url_key)

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
    scopes = _normalize_scopes(
        tokens.get("scope") or prior.get("scopes") or prior.get("scope")
    )
    stored_at = _resolve_stored_at(tokens, prior)

    record: dict[str, Any] = {
        "provider": "linear",
        "provider_host": "linear.app",
        "auth_type": "oauth",
        "token_storage": "file",
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
    _apply_optional_fields(
        record, tokens, prior, ("expires_at", "expires_in", "cloud_id")
    )
    workspaces = prior.get("workspaces")
    if isinstance(workspaces, dict) and workspaces:
        record["workspaces"] = workspaces
    organizations = prior.get("organizations")
    if isinstance(organizations, dict) and organizations:
        record["organizations"] = organizations
    active_org = prior.get("active_organization_id")
    if active_org:
        record["active_organization_id"] = active_org
    return record


def linear_workspaces_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    workspaces = entry.get("workspaces")
    if isinstance(workspaces, dict):
        return workspaces
    return {}


def build_product_integration_record(
    product: str, credentials: dict[str, Any]
) -> dict[str, Any]:
    """Build metadata for a single Atlassian product (jira or confluence)."""
    record = build_atlassian_integration_record(credentials)
    record["provider"] = product.strip().lower()
    return record


def build_bitbucket_integration_record(credentials: dict[str, Any]) -> dict[str, Any]:
    now = utc_now_iso()
    email = str(credentials.get("email") or "").strip()
    account_name = str(credentials.get("account_name") or "").strip()
    workspaces = credentials.get("workspaces")
    record: dict[str, Any] = {
        "provider": "bitbucket",
        "provider_host": "bitbucket.org",
        "site_url": "https://bitbucket.org",
        "auth_type": "api_token",
        "token_storage": "keychain",
        "stored_at": _stored_at_from_credentials(credentials),
        "created_at": credentials.get("created_at") or now,
        "updated_at": now,
        "metadata": {"auth_flow": "api_token"},
    }
    if email or account_name:
        account: dict[str, Any] = {}
        if email:
            account["email"] = email
            record["email"] = email
        if account_name:
            account["name"] = account_name
            record["account_name"] = account_name
        record["account"] = account
    if isinstance(workspaces, dict):
        record["workspaces"] = dict(workspaces)
    return record


def _stored_at_from_credentials(credentials: dict[str, Any]) -> Any:
    stored_at = credentials.get("stored_at")
    return time.time() if stored_at is None else stored_at


def _atlassian_site_metadata(
    cloud_id: str,
    site_url: str,
    site_name: str,
) -> dict[str, Any]:
    site: dict[str, Any] = {}
    if cloud_id:
        site["cloud_id"] = cloud_id
    if site_url:
        site["site_url"] = site_url
    if site_name:
        site["site_name"] = site_name
    return site


def _apply_atlassian_identity_fields(
    record: dict[str, Any],
    *,
    email: str,
    cloud_id: str,
    site_url: str,
    site_name: str,
    account: dict[str, Any],
    site: dict[str, Any],
) -> None:
    if account:
        record["account"] = account
    if site:
        record["site"] = site
    for key, value in (
        ("email", email),
        ("cloud_id", cloud_id),
        ("site_url", site_url),
        ("site_name", site_name),
    ):
        if value:
            record[key] = value


def build_atlassian_integration_record(credentials: dict[str, Any]) -> dict[str, Any]:
    """Build non-secret Atlassian metadata with nested account/site plus flat fields."""
    now = utc_now_iso()
    email = str(credentials.get("email") or "").strip()
    cloud_id = str(credentials.get("cloud_id") or "").strip()
    site_url = str(credentials.get("site_url") or "").strip()
    site_name = str(credentials.get("site_name") or "").strip()
    account = {"email": email} if email else {}
    site = _atlassian_site_metadata(cloud_id, site_url, site_name)
    token_style = str(credentials.get("token_style") or "classic").strip() or "classic"

    record: dict[str, Any] = {
        "provider": "atlassian",
        "provider_host": "atlassian.net",
        "auth_type": "api_token",
        "token_style": token_style,
        "token_storage": "file",
        "stored_at": _stored_at_from_credentials(credentials),
        "created_at": credentials.get("created_at") or now,
        "updated_at": now,
        "metadata": {"auth_flow": "classic_api_token"},
    }
    _apply_atlassian_identity_fields(
        record,
        email=email,
        cloud_id=cloud_id,
        site_url=site_url,
        site_name=site_name,
        account=account,
        site=site,
    )
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
