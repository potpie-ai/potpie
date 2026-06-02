"""Atlassian Cloud API token authentication for Jira and Confluence CLI integrations."""

from __future__ import annotations

import base64
import enum
import re
import sys
import time
import webbrowser
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import typer

from adapters.inbound.cli.credentials_store import (
    ProviderCredentialError,
    credentials_path,
    get_confluence_credentials,
    get_jira_credentials,
    save_confluence_credentials,
    save_jira_credentials,
)
from adapters.inbound.cli.output import emit_error, print_plain_line
from adapters.inbound.cli.provider_config import (
    ATLASSIAN_ACCESSIBLE_RESOURCES_URL,
    ATLASSIAN_API_TOKEN_PAGE,
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
        return str(
            data.get("displayName") or data.get("emailAddress") or ""
        ).strip()
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

    with httpx.Client(timeout=_SITE_PROBE_TIMEOUT) as client:
        for path in _gateway_probe_paths(product):
            url = f"{base}{path}"
            for scheme, headers in _auth_header_variants(email, api_token):
                try:
                    response = client.get(url, headers=headers)
                except httpx.HTTPError:
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

    if saw_403 and last_kind != AtlassianAuthErrorKind.INVALID_CREDENTIALS:
        last_kind = AtlassianAuthErrorKind.INSUFFICIENT_SCOPES

    return AtlassianVerifyResult(
        ok=False,
        error_kind=last_kind,
        http_status=last_status,
    )


def fetch_cloud_id_for_site(site_url: str) -> str:
    normalized = normalize_site_url(site_url)
    if not normalized:
        return ""
    url = f"{normalized}/_edge/tenant_info"
    try:
        with httpx.Client(timeout=_SITE_PROBE_TIMEOUT) as client:
            response = client.get(url, headers={"Accept": "application/json"})
    except httpx.HTTPError:
        return ""
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
) -> list[dict[str, Any]]:
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        for _scheme, headers in _auth_header_variants(email, api_token):
            try:
                response = client.get(
                    ATLASSIAN_ACCESSIBLE_RESOURCES_URL,
                    headers=headers,
                )
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                sites = _parse_accessible_resources(response.json())
            except ValueError:
                continue
            if sites:
                return sites
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
    slug = urlparse(normalized).netloc.removesuffix(".atlassian.net") if normalized else ""
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
        name = verify.display_name or candidate.get("site_name") or candidate["site_url"]
        found.append(
            _site_record(
                site_url=candidate["site_url"],
                cloud_id=cloud_id,
                site_name=str(name),
            )
        )
    return found


def _prompt_site_subdomain() -> str:
    return typer.prompt(
        "Enter your Atlassian site subdomain "
        "(e.g. 'potpie-team' for potpie-team.atlassian.net)"
    ).strip()


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


def _prompt_and_resolve_site() -> tuple[dict[str, Any] | None, AtlassianAuthErrorKind | None]:
    """Prompt once for site subdomain and resolve cloud ID."""
    return _resolve_site_from_subdomain(_prompt_site_subdomain())


def _cli_credentials_supplied(
    email: str | None,
    api_token: str | None,
    site_subdomain: str | None,
) -> bool:
    return bool(
        (email or "").strip()
        and (api_token or "").strip()
        and (site_subdomain or "").strip()
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


def _prompt_credentials() -> tuple[str, str]:
    email = typer.prompt("Enter your Atlassian email").strip()
    if not email:
        raise typer.Exit(code=1)
    api_token = typer.prompt("Enter your API token", hide_input=True).strip()
    if not api_token:
        raise typer.Exit(code=1)
    return email, api_token


def _auth_failure_message(
    product: AtlassianProduct,
    error_kind: AtlassianAuthErrorKind | None = None,
) -> str:
    name = product.capitalize()
    lines = [
        f"Could not authenticate {name} with Atlassian.",
        "  - Use an API token from id.atlassian.com (scoped tokens are supported)",
        "  - Email must match the Atlassian account that created the token",
        "  - Your account must have access to the product on the selected site",
    ]
    lines.append(
        "  - Use Create API token (without scopes) for Jira + Confluence on one token"
    )
    lines.append(
        "  - Or use scoped tokens: Jira needs read:jira-work; "
        "Confluence needs read/content scopes (one product per token)"
    )
    if error_kind == AtlassianAuthErrorKind.INVALID_CREDENTIALS:
        lines.insert(1, "  - Invalid email or API token")
    elif error_kind == AtlassianAuthErrorKind.INSUFFICIENT_SCOPES:
        if product == "confluence":
            scope_hint = (
                "  - Token is missing required read scopes "
                "(Confluence: read:confluence-content at minimum)"
            )
        else:
            scope_hint = (
                "  - Token is missing required read scopes "
                "(Jira: read:jira-work at minimum)"
            )
        lines.insert(1, scope_hint)
    elif error_kind == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED:
        lines.insert(1, "  - Could not resolve cloud ID for the site")
    elif error_kind == AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED:
        lines.insert(
            1,
            f"  - Token or account cannot access {name} on this site",
        )
    else:
        lines.insert(
            1,
            "  - Check the site subdomain from your Jira URL (e.g. acme.atlassian.net)",
        )
    return "\n".join(lines)


def _get_product_credentials(product: AtlassianProduct) -> dict[str, Any]:
    if product == "jira":
        return get_jira_credentials()
    return get_confluence_credentials()


def _save_product_credentials(product: AtlassianProduct, payload: dict[str, Any]) -> None:
    if product == "jira":
        save_jira_credentials(payload)
    else:
        save_confluence_credentials(payload)


def run_atlassian_api_token_auth(
    product: AtlassianProduct,
    *,
    force: bool = False,
    as_json: bool = False,
    verbose: bool = False,
    email: str | None = None,
    api_token: str | None = None,
    site_subdomain: str | None = None,
) -> None:
    """Authenticate Jira or Confluence using an Atlassian API token (one product only)."""
    product_label = product.capitalize()
    try:
        existing = _get_product_credentials(product)
    except ProviderCredentialError as exc:
        emit_error(f"{product_label} credential lookup failed", str(exc), verbose=verbose)
        existing = {}

    if existing.get("api_token") and existing.get("site_url") and not force:
        print_plain_line(
            f"{product_label} is already connected.",
            as_json=as_json,
            json_payload={
                "ok": True,
                "already_connected": True,
                "provider": product,
                "site_url": existing.get("site_url"),
                "site_name": existing.get("site_name"),
                "cloud_id": existing.get("cloud_id"),
            },
        )
        return

    supplied = _cli_credentials_supplied(email, api_token, site_subdomain)
    if not sys.stdin.isatty() and not supplied:
        emit_error(
            f"{product_label} authentication requires a terminal",
            "Run in an interactive shell to enter email, API token, and site subdomain, "
            "or pass all three when invoking login non-interactively.",
            verbose=verbose,
        )
        raise typer.Exit(code=1)

    if supplied:
        email_value = (email or "").strip()
        api_token_value = (api_token or "").strip()
        site, last_error = _resolve_site_from_subdomain(site_subdomain or "")
    else:
        if not as_json:
            print_plain_line("Opening Atlassian API token page...", as_json=False)
            print_plain_line(
                f"If the browser does not open, visit:\n{ATLASSIAN_API_TOKEN_PAGE}",
                as_json=False,
            )
            print_plain_line(
                f"Create an API token (without scopes) with access to {product_label}.",
                as_json=False,
            )
            print_plain_line(
                "At id.atlassian.com choose Create API token — not Create API token with scopes.",
                as_json=False,
            )
        webbrowser.open(ATLASSIAN_API_TOKEN_PAGE, new=1)
        email_value, api_token_value = _prompt_credentials()
        site, last_error = _prompt_and_resolve_site()
    email, api_token = email_value, api_token_value
    if not site:
        emit_error(
            f"{product.capitalize()} authentication failed",
            _auth_failure_message(product, last_error),
            verbose=verbose,
        )
        raise typer.Exit(code=1)

    site, last_error = _finalize_selected_site(email, api_token, site, product)
    if not site:
        emit_error(
            f"{product_label} authentication failed",
            _auth_failure_message(product, last_error),
            verbose=verbose,
        )
        raise typer.Exit(code=1)

    token_style = str(site.get("token_style") or "").strip() or "classic"
    payload = {
        "auth_type": "api_token",
        "token_style": token_style,
        "email": email,
        "api_token": api_token,
        "cloud_id": str(site.get("cloud_id") or "").strip(),
        "site_url": site["site_url"],
        "site_name": site["site_name"],
        "stored_at": time.time(),
    }
    try:
        _save_product_credentials(product, payload)
    except ProviderCredentialError as exc:
        emit_error(f"{product_label} credential storage failed", str(exc), verbose=verbose)
        raise typer.Exit(code=1) from exc

    summary = (
        f"Connected {product_label} to {site['site_url']}. "
        f"Stored tokens in system keychain; metadata saved to {credentials_path()}."
    )
    print_plain_line(
        summary,
        as_json=as_json,
        json_payload={
            "ok": True,
            "provider": product,
            "token_style": token_style,
            "site_url": site["site_url"],
            "site_name": site["site_name"],
            "cloud_id": payload["cloud_id"],
            "path": str(credentials_path()),
            "token_storage": "keychain",
            "product_verified": product,
        },
    )


def fetch_accessible_resources(email: str, api_token: str) -> list[dict[str, Any]]:
    """Backwards-compatible alias for tests."""
    return discover_sites_with_api_token(email, api_token, "jira")
