"""Outbound GitLab client: PAT verification + instance URL normalization.

Pure HTTP/transport for GitLab (no CLI/presentation). The interactive
login command lives in ``adapters.inbound.cli.auth.gitlab_auth``.
"""

from __future__ import annotations

import enum
import os
from typing import Any
from urllib.parse import urlparse

from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.provider_config import (
    GITLAB_DEFAULT_INSTANCE,
    gitlab_api_base_url,
)


_HTTP_TIMEOUT = 30.0

# GitLab SaaS is always served from the host root (no subpath install).
_KNOWN_GITLAB_SAAS_HOSTS = frozenset({"gitlab.com", "www.gitlab.com"})

# Common mount prefixes when GitLab is installed under a URL path.
_GITLAB_SUBPATH_MOUNTS = frozenset({"gitlab", "git", "gl", "scm"})


def _instance_path_prefix(path: str, *, host: str) -> str:
    """Return a subpath mount prefix, or empty when path is not the instance root."""
    normalized_path = (path or "").rstrip("/")
    if not normalized_path:
        return ""

    hostname = host.lower().split(":")[0]
    if hostname in _KNOWN_GITLAB_SAAS_HOSTS:
        return ""

    if normalized_path.startswith(("/-/", "/groups/", "/projects/")):
        return ""

    segments = [segment for segment in normalized_path.split("/") if segment]
    if not segments:
        return ""

    if len(segments) == 1:
        return f"/{segments[0]}"

    if segments[0].lower() in _GITLAB_SUBPATH_MOUNTS:
        return f"/{segments[0]}"

    return ""


def normalize_instance_url(url: str, *, allow_http: bool | None = None) -> str:
    """Normalize a GitLab instance URL to ``https://host[:port][/path]``."""
    value = url.strip().rstrip("/")
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    parsed = urlparse(value)
    if not parsed.netloc:
        return ""
    if parsed.scheme == "http":
        if allow_http is None:
            allow_http = os.environ.get(
                "POTPIE_GITLAB_ALLOW_HTTP", ""
            ).strip().lower() in (
                "1",
                "true",
                "yes",
            )
        scheme = "http" if allow_http else "https"
    else:
        scheme = parsed.scheme or "https"
    base = f"{scheme}://{parsed.netloc}"
    path_prefix = _instance_path_prefix(parsed.path or "", host=parsed.netloc)
    if path_prefix:
        return f"{base}{path_prefix}"
    return base


class GitLabAuthErrorKind(enum.StrEnum):
    INVALID_CREDENTIALS = "invalid_credentials"
    INSUFFICIENT_SCOPES = "insufficient_scopes"
    INSTANCE_UNREACHABLE = "instance_unreachable"
    UNKNOWN = "unknown"


def instance_host(instance_url: str) -> str:
    """Extract the host (with optional port) from a normalized instance URL."""
    normalized = normalize_instance_url(instance_url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    return parsed.netloc or ""


def gitlab_auth_headers(pat: str) -> dict[str, str]:
    return {
        "PRIVATE-TOKEN": pat.strip(),
        "Accept": "application/json",
    }


def _classify_status(status: int) -> GitLabAuthErrorKind:
    if status == 401:
        return GitLabAuthErrorKind.INVALID_CREDENTIALS
    if status == 403:
        return GitLabAuthErrorKind.INSUFFICIENT_SCOPES
    return GitLabAuthErrorKind.UNKNOWN


def verify_instance_access(
    instance_url: str,
    pat: str,
    *,
    http: HttpClient | None = None,
) -> tuple[bool, GitLabAuthErrorKind | None, dict[str, Any]]:
    """Verify PAT against ``GET /api/v4/user``.

    Returns ``(ok, error_kind, user_data)``.
    """
    normalized = normalize_instance_url(instance_url) or GITLAB_DEFAULT_INSTANCE
    api_base = gitlab_api_base_url(normalized)
    headers = gitlab_auth_headers(pat)

    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        try:
            response = http.get(f"{api_base}/user", headers=headers)
        except AuthHttpError:
            return False, GitLabAuthErrorKind.INSTANCE_UNREACHABLE, {}
        if response.status_code != 200:
            return False, _classify_status(response.status_code), {}
        try:
            data = response.json()
        except ValueError:
            return False, GitLabAuthErrorKind.UNKNOWN, {}
        if not isinstance(data, dict):
            return False, GitLabAuthErrorKind.UNKNOWN, {}
        return True, None, data
    finally:
        if owns:
            http.close()


def verify_read_api_scope(
    instance_url: str,
    pat: str,
    *,
    http: HttpClient | None = None,
) -> tuple[bool, GitLabAuthErrorKind | None]:
    """Confirm the PAT has at least ``read_api`` scope by listing one project."""
    normalized = normalize_instance_url(instance_url) or GITLAB_DEFAULT_INSTANCE
    api_base = gitlab_api_base_url(normalized)
    headers = gitlab_auth_headers(pat)

    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        try:
            response = http.get(
                f"{api_base}/projects",
                headers=headers,
                params={"membership": "true", "per_page": "1"},
            )
        except AuthHttpError:
            return False, GitLabAuthErrorKind.INSTANCE_UNREACHABLE
        if response.status_code == 200:
            return True, None
        return False, _classify_status(response.status_code)
    finally:
        if owns:
            http.close()


def parse_user_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Extract account metadata from a ``/api/v4/user`` response."""
    account: dict[str, Any] = {}
    user_id = data.get("id")
    if user_id is not None:
        account["id"] = str(user_id)
    username = data.get("username")
    if username:
        account["username"] = str(username)
    name = data.get("name")
    if name:
        account["name"] = str(name)
    email = data.get("email")
    if email:
        account["email"] = str(email)
    web_url = data.get("web_url")
    if web_url:
        account["web_url"] = str(web_url)
    return account
