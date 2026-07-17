"""Outbound Atlassian Cloud client: API-token verification + site discovery.

Pure HTTP/transport for Jira & Confluence (no CLI/presentation). The interactive
login command lives in ``potpie_context_engine.adapters.inbound.cli.auth.atlassian_auth``.
"""

from __future__ import annotations

import base64
import enum
import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from potpie_context_engine.adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from potpie_context_engine.adapters.outbound.cli_auth.provider_config import (
    ATLASSIAN_ACCESSIBLE_RESOURCES_URL,
    AtlassianProduct,
    atlassian_confluence_gateway_url,
    atlassian_jira_gateway_url,
)


_HTTP_TIMEOUT = 30.0


_SITE_PROBE_TIMEOUT = 15.0


_JIRA_GATEWAY_PROBE_PATHS = (
    "/rest/api/3/project/search?maxResults=1",
    "/rest/api/3/myself",
)


_CONFLUENCE_GATEWAY_PROBE_PATHS = (
    "/wiki/rest/api/space?limit=1",
    "/wiki/rest/api/user/current",
)


class AtlassianAuthErrorKind(enum.StrEnum):
    INVALID_CREDENTIALS = "invalid_credentials"
    INSUFFICIENT_SCOPES = "insufficient_scopes"
    SITE_DISCOVERY_FAILED = "site_discovery_failed"
    PRODUCT_ACCESS_DENIED = "product_access_denied"
    UNKNOWN = "unknown"


AtlassianAuthScheme = Literal["basic", "bearer"]


@dataclass(frozen=True)
class AtlassianVerifyResult:
    ok: bool
    error_kind: AtlassianAuthErrorKind | None = None
    http_status: int | None = None
    display_name: str = ""
    succeeded_scheme: AtlassianAuthScheme | None = None


def token_style_from_succeeded_scheme(scheme: AtlassianAuthScheme | None) -> str:
    """Map gateway auth scheme to stored token_style metadata."""
    if scheme == "bearer":
        return "bearer"
    return "classic"


def atlassian_basic_auth_header(email: str, api_token: str) -> str:
    raw = f"{email.strip()}:{api_token.strip()}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def atlassian_bearer_auth_header(api_token: str) -> str:
    return f"Bearer {api_token.strip()}"


def _auth_header_variants(
    email: str, api_token: str
) -> list[tuple[AtlassianAuthScheme, dict[str, str]]]:
    accept = {"Accept": "application/json"}
    return [
        (
            "basic",
            {
                **accept,
                "Authorization": atlassian_basic_auth_header(email, api_token),
            },
        ),
        (
            "bearer",
            {
                **accept,
                "Authorization": atlassian_bearer_auth_header(api_token),
            },
        ),
    ]


def normalize_site_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    parsed = urlparse(value)
    if not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def site_url_from_subdomain(subdomain: str) -> str:
    slug = subdomain.strip().lower().removesuffix(".atlassian.net")
    slug = slug.removeprefix("https://").removeprefix("http://")
    if not slug or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        return ""
    return f"https://{slug}.atlassian.net"


def _gateway_base_url(cloud_id: str, product: AtlassianProduct) -> str:
    if product == "jira":
        return atlassian_jira_gateway_url(cloud_id)
    return atlassian_confluence_gateway_url(cloud_id)


def _gateway_probe_paths(product: AtlassianProduct) -> tuple[str, ...]:
    if product == "jira":
        return _JIRA_GATEWAY_PROBE_PATHS
    return _CONFLUENCE_GATEWAY_PROBE_PATHS


def _classify_gateway_status(status: int) -> AtlassianAuthErrorKind:
    if status == 401:
        return AtlassianAuthErrorKind.INVALID_CREDENTIALS
    if status == 403:
        return AtlassianAuthErrorKind.INSUFFICIENT_SCOPES
    if status == 404:
        return AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED
    return AtlassianAuthErrorKind.UNKNOWN


def _parse_profile_name(data: Any, *, product: AtlassianProduct) -> str:
    if not isinstance(data, dict):
        return ""
    if product == "jira":
        return str(data.get("displayName") or data.get("emailAddress") or "").strip()
    return str(data.get("displayName") or data.get("username") or "").strip()


def _parse_gateway_probe_success(
    data: Any,
    *,
    product: AtlassianProduct,
    path: str,
) -> str:
    if product == "jira" and "myself" in path:
        return _parse_profile_name(data, product=product)
    if product == "jira" and isinstance(data, dict):
        values = data.get("values")
        if isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, dict):
                return str(first.get("name") or first.get("key") or "").strip()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return str(data[0].get("name") or data[0].get("key") or "").strip()
    if product == "confluence" and "user/current" in path:
        return _parse_profile_name(data, product=product)
    if product == "confluence" and isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                return str(first.get("name") or first.get("key") or "").strip()
    return _parse_profile_name(data, product=product)


def verify_gateway_product(
    email: str,
    api_token: str,
    cloud_id: str,
    product: AtlassianProduct,
    *,
    http: HttpClient | None = None,
) -> AtlassianVerifyResult:
    """Verify credentials against the Atlassian scoped-token gateway."""
    cloud_id = cloud_id.strip()
    if not cloud_id:
        return AtlassianVerifyResult(
            ok=False,
            error_kind=AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED,
        )

    base = _gateway_base_url(cloud_id, product)
    last_status: int | None = None
    last_kind = AtlassianAuthErrorKind.UNKNOWN
    saw_403 = False

    owns = http is None
    http = http or AuthHttpClient(timeout=_SITE_PROBE_TIMEOUT)
    try:
        for path in _gateway_probe_paths(product):
            url = f"{base}{path}"
            for scheme, headers in _auth_header_variants(email, api_token):
                try:
                    response = http.get(url, headers=headers)
                except AuthHttpError:
                    last_kind = AtlassianAuthErrorKind.UNKNOWN
                    last_status = None
                    continue
                last_status = response.status_code
                if response.status_code == 200:
                    payload = response.json() if response.content else {}
                    return AtlassianVerifyResult(
                        ok=True,
                        display_name=_parse_gateway_probe_success(
                            payload,
                            product=product,
                            path=path,
                        ),
                        succeeded_scheme=scheme,
                    )
                kind = _classify_gateway_status(response.status_code)
                last_kind = kind
                if response.status_code == 403:
                    saw_403 = True
                if response.status_code == 401:
                    continue
                break
    finally:
        if owns:
            http.close()

    if saw_403 and last_kind != AtlassianAuthErrorKind.INVALID_CREDENTIALS:
        last_kind = AtlassianAuthErrorKind.INSUFFICIENT_SCOPES

    return AtlassianVerifyResult(
        ok=False,
        error_kind=last_kind,
        http_status=last_status,
    )


def fetch_cloud_id_for_site(site_url: str, *, http: HttpClient | None = None) -> str:
    normalized = normalize_site_url(site_url)
    if not normalized:
        return ""
    url = f"{normalized}/_edge/tenant_info"
    owns = http is None
    http = http or AuthHttpClient(timeout=_SITE_PROBE_TIMEOUT)
    try:
        response = http.get(url, headers={"Accept": "application/json"})
    except AuthHttpError:
        return ""
    finally:
        if owns:
            http.close()
    if response.status_code != 200:
        return ""
    data = response.json()
    if isinstance(data, dict):
        cloud_id = data.get("cloudId") or data.get("cloud_id")
        return str(cloud_id).strip() if cloud_id else ""
    return ""


def _parse_accessible_resources(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    sites: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        site_url = normalize_site_url(str(item.get("url") or ""))
        cloud_id = str(item.get("id") or "").strip()
        if site_url and cloud_id:
            sites.append(
                {
                    "cloud_id": cloud_id,
                    "site_url": site_url,
                    "site_name": str(item.get("name") or site_url).strip(),
                }
            )
    return sites


def _fetch_accessible_resources(
    email: str,
    api_token: str,
    *,
    http: HttpClient | None = None,
) -> list[dict[str, Any]]:
    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        for _scheme, headers in _auth_header_variants(email, api_token):
            try:
                response = http.get(
                    ATLASSIAN_ACCESSIBLE_RESOURCES_URL,
                    headers=headers,
                )
            except AuthHttpError:
                continue
            if response.status_code != 200:
                continue
            try:
                sites = _parse_accessible_resources(response.json())
            except ValueError:
                continue
            if sites:
                return sites
    finally:
        if owns:
            http.close()
    return []


def _merge_site_candidate(
    candidates: list[dict[str, Any]],
    seen_urls: set[str],
    *,
    site_url: str,
    cloud_id: str = "",
    site_name: str = "",
    source: str = "",
) -> None:
    normalized = normalize_site_url(site_url)
    if not normalized or normalized in seen_urls:
        return
    cid = cloud_id.strip() or fetch_cloud_id_for_site(normalized)
    if not cid:
        return
    seen_urls.add(normalized)
    slug = urlparse(normalized).netloc.removesuffix(".atlassian.net")
    candidates.append(
        {
            "cloud_id": cid,
            "site_url": normalized,
            "site_name": (site_name or slug or normalized).strip(),
            "source": source,
        }
    )


def _site_record(
    *,
    site_url: str,
    cloud_id: str,
    site_name: str = "",
) -> dict[str, Any]:
    normalized = normalize_site_url(site_url)
    slug = (
        urlparse(normalized).netloc.removesuffix(".atlassian.net") if normalized else ""
    )
    return {
        "cloud_id": cloud_id,
        "site_url": normalized,
        "site_name": site_name or slug or normalized,
    }


def verify_site_with_api_token(
    email: str,
    api_token: str,
    site_url: str,
    product: AtlassianProduct = "jira",
) -> dict[str, Any] | None:
    """Resolve cloudId and verify API token access via the Atlassian gateway."""
    normalized = normalize_site_url(site_url)
    if not normalized:
        return None

    cloud_id = fetch_cloud_id_for_site(normalized)
    if not cloud_id:
        return None

    result = verify_gateway_product(email, api_token, cloud_id, product)
    if not result.ok:
        return None

    slug = urlparse(normalized).netloc.removesuffix(".atlassian.net")
    site_name = result.display_name or slug or normalized
    return _site_record(
        site_url=normalized,
        cloud_id=cloud_id,
        site_name=site_name,
    )


def _candidate_subdomains_from_email(email: str) -> list[str]:
    local, _, domain = email.strip().lower().partition("@")
    candidates: list[str] = []
    if domain:
        org = domain.split(".", 1)[0]
        if org and org not in {"gmail", "yahoo", "hotmail", "outlook", "icloud"}:
            candidates.append(org)
    if local and local not in candidates:
        candidates.append(local)
    return candidates


def collect_site_candidates(email: str, api_token: str) -> list[dict[str, Any]]:
    """List plausible Atlassian Cloud sites for this account."""
    return collect_login_site_candidates(email, api_token, include_email_hints=True)


def collect_login_site_candidates(
    email: str,
    api_token: str,
    *,
    include_email_hints: bool = False,
) -> list[dict[str, Any]]:
    """Sites for login: Atlassian accessible-resources (optional email-domain hints)."""
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for site in _fetch_accessible_resources(email, api_token):
        _merge_site_candidate(
            candidates,
            seen_urls,
            site_url=str(site.get("site_url") or ""),
            cloud_id=str(site.get("cloud_id") or ""),
            site_name=str(site.get("site_name") or ""),
            source="accessible-resources",
        )

    if include_email_hints:
        for subdomain in _candidate_subdomains_from_email(email):
            site_url = site_url_from_subdomain(subdomain)
            if site_url:
                _merge_site_candidate(
                    candidates,
                    seen_urls,
                    site_url=site_url,
                    site_name=subdomain,
                    source="email-hint",
                )

    return candidates


def _finalize_atlassian_site_unscoped(
    email: str,
    api_token: str,
    site: dict[str, Any],
) -> tuple[dict[str, Any] | None, AtlassianAuthErrorKind | None]:
    """Verify Jira and Confluence gateway access for a classic (unscoped) API token."""
    site, err = _finalize_selected_site(email, api_token, site, "jira")
    if not site:
        return None, err
    cloud_id = str(site.get("cloud_id") or "").strip()
    confluence = verify_gateway_product(email, api_token, cloud_id, "confluence")
    if not confluence.ok:
        return None, confluence.error_kind
    return site, None


def discover_sites_with_api_token(
    email: str,
    api_token: str,
    product: AtlassianProduct = "jira",
) -> list[dict[str, Any]]:
    """Return site candidates that pass gateway verification for the given product."""
    found: list[dict[str, Any]] = []
    for candidate in collect_site_candidates(email, api_token):
        cloud_id = str(candidate.get("cloud_id") or "").strip()
        if not cloud_id:
            continue
        verify = verify_gateway_product(email, api_token, cloud_id, product)
        if not verify.ok:
            continue
        name = (
            verify.display_name or candidate.get("site_name") or candidate["site_url"]
        )
        found.append(
            _site_record(
                site_url=candidate["site_url"],
                cloud_id=cloud_id,
                site_name=str(name),
            )
        )
    return found


def _resolve_site_from_subdomain(
    subdomain: str,
) -> tuple[dict[str, Any] | None, AtlassianAuthErrorKind | None]:
    """Resolve cloud ID from an Atlassian site subdomain (no prompt)."""
    subdomain = subdomain.strip()
    if not subdomain:
        return None, AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED
    site_url = site_url_from_subdomain(subdomain)
    if not site_url:
        return None, AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED
    cloud_id = fetch_cloud_id_for_site(site_url)
    if not cloud_id:
        return None, AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED
    slug = urlparse(site_url).netloc.removesuffix(".atlassian.net")
    return (
        _site_record(
            site_url=site_url,
            cloud_id=cloud_id,
            site_name=slug or site_url,
        ),
        None,
    )


def _finalize_selected_site(
    email: str,
    api_token: str,
    site: dict[str, Any],
    product: AtlassianProduct,
) -> tuple[dict[str, Any] | None, AtlassianAuthErrorKind | None]:
    cloud_id = str(site.get("cloud_id") or "").strip()
    if not cloud_id:
        cloud_id = fetch_cloud_id_for_site(str(site.get("site_url") or ""))
    if not cloud_id:
        return None, AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED

    result = verify_gateway_product(email, api_token, cloud_id, product)
    if not result.ok:
        return None, result.error_kind

    site = {
        **site,
        "cloud_id": cloud_id,
        "token_style": token_style_from_succeeded_scheme(result.succeeded_scheme),
    }
    return site, None


def fetch_accessible_resources(email: str, api_token: str) -> list[dict[str, Any]]:
    """Backwards-compatible alias for tests."""
    return discover_sites_with_api_token(email, api_token, "jira")
