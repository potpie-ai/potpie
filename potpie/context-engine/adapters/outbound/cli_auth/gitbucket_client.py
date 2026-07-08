"""Outbound GitBucket HTTP client: PAT verification + URL normalization.

Pure HTTP/transport for GitBucket (no CLI/presentation). The interactive
login command lives in ``adapters.inbound.cli.auth.gitbucket_commands``.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.provider_config import (
    GITBUCKET_ALLOW_INSECURE_HTTP_ENV_VARS,
    GITBUCKET_API_VERSION,
    GITBUCKET_TOKEN_PAGE_SUFFIX,
)

_HTTP_TIMEOUT = 30.0


class GitBucketClientError(Exception):
    """Failed to communicate with a GitBucket instance."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class GitBucketAccount:
    login: str
    email: str = ""
    site_admin: bool = False
    html_url: str = ""


def normalize_gitbucket_host_url(host_url: str) -> str:
    """Normalize a GitBucket host URL: strip trailing slashes, ensure scheme.

    Handles subpath deployments (e.g. ``https://git.company.com/gitbucket``).
    Bare hostnames are assumed HTTPS.
    """
    url = (host_url or "").strip().rstrip("/")
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed = urlparse(url)
    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )
    return normalized


def _gitbucket_allow_insecure_http() -> bool:
    for name in GITBUCKET_ALLOW_INSECURE_HTTP_ENV_VARS:
        value = os.getenv(name, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


def _is_loopback_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    host = hostname.strip().lower()
    if host == "localhost":
        return True
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def ensure_gitbucket_pat_transport_allowed(host_url: str) -> None:
    """Reject plain HTTP to non-loopback hosts before sending a PAT."""
    normalized = normalize_gitbucket_host_url(host_url)
    if not normalized:
        return
    parsed = urlparse(normalized)
    if parsed.scheme != "http":
        return
    if _is_loopback_hostname(parsed.hostname):
        return
    if _gitbucket_allow_insecure_http():
        return
    raise GitBucketClientError(
        "GitBucket over plain HTTP is only allowed for localhost. "
        "Use https:// or set POTPIE_GITBUCKET_ALLOW_INSECURE_HTTP=1 for development."
    )


def gitbucket_api_base(host_url: str) -> str:
    """Return the GitBucket REST API base URL for a given host URL."""
    return f"{normalize_gitbucket_host_url(host_url)}/api/{GITBUCKET_API_VERSION}"


def gitbucket_token_page_url(host_url: str, login: str) -> str:
    """Return the GitBucket Applications page where users create personal access tokens."""
    normalized = normalize_gitbucket_host_url(host_url)
    username = login.strip().strip("/")
    if not username:
        raise ValueError("GitBucket username is required for the token page URL.")
    return f"{normalized}/{username}/{GITBUCKET_TOKEN_PAGE_SUFFIX}"


def _token_auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"token {token.strip()}"}


def verify_gitbucket_token(
    host_url: str,
    token: str,
    *,
    http: HttpClient | None = None,
) -> GitBucketAccount:
    """Verify a GitBucket PAT by calling ``GET /api/v3/user``.

    Returns a :class:`GitBucketAccount` on success.
    Raises :class:`GitBucketClientError` on failure.
    """
    normalized = normalize_gitbucket_host_url(host_url)
    ensure_gitbucket_pat_transport_allowed(normalized)
    api_base = gitbucket_api_base(normalized)
    headers = _token_auth_header(token)
    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        response = http.get(f"{api_base}/user", headers=headers)
    except AuthHttpError as exc:
        raise GitBucketClientError(
            f"Could not reach GitBucket at {normalized}: {exc}"
        ) from exc
    finally:
        if owns:
            http.close()

    if response.status_code == 401:
        raise GitBucketClientError(
            "Authentication failed. Check that your personal access token is correct.",
            status_code=401,
        )
    if response.status_code == 404:
        raise GitBucketClientError(
            f"GitBucket API not found at {api_base}. "
            "Verify the host URL. If GitBucket is deployed at a subpath "
            "(e.g. https://git.company.com/gitbucket), include the full path.",
            status_code=404,
        )
    if response.status_code != 200:
        raise GitBucketClientError(
            f"GitBucket API returned HTTP {response.status_code}.",
            status_code=response.status_code,
        )

    try:
        data: Any = response.json()
    except ValueError as exc:
        raise GitBucketClientError(
            "GitBucket API returned a non-JSON response."
        ) from exc

    if not isinstance(data, dict):
        raise GitBucketClientError(
            "GitBucket API returned an unexpected response format."
        )

    login = str(data.get("login") or "").strip()
    if not login:
        raise GitBucketClientError("GitBucket API response missing 'login' field.")
    email = str(data.get("email") or "").strip()
    site_admin = bool(data.get("site_admin"))
    html_url = str(data.get("html_url") or "").strip()
    return GitBucketAccount(
        login=login,
        email=email,
        site_admin=site_admin,
        html_url=html_url,
    )


__all__ = [
    "GitBucketAccount",
    "GitBucketClientError",
    "ensure_gitbucket_pat_transport_allowed",
    "gitbucket_api_base",
    "gitbucket_token_page_url",
    "normalize_gitbucket_host_url",
    "verify_gitbucket_token",
]
