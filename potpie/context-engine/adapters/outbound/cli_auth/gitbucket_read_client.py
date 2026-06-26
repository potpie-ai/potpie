"""Read-only GitBucket HTTP client.

Pure transport + parsing for GitBucket reads (repos, issues, pull requests) —
no CLI/presentation. The interactive CLI commands live in
``adapters.inbound.cli.auth.gitbucket_commands``.
"""

from __future__ import annotations

from typing import Any

from adapters.outbound.cli_auth.credentials_store import (
    ProviderCredentialError,
    get_gitbucket_credentials,
)
from adapters.outbound.cli_auth.errors import CliAuthError
from adapters.outbound.cli_auth.gitbucket_client import (
    _token_auth_header,
    gitbucket_api_base,
)
from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient

_HTTP_TIMEOUT = 30.0
_DEFAULT_REPO_LIMIT = 30


class GitBucketReadError(CliAuthError):
    """Failed to read GitBucket data with stored credentials."""


def _load_credentials() -> tuple[str, str]:
    """Return ``(host_url, token)`` from stored credentials or raise."""
    credentials = get_gitbucket_credentials()
    host_url = str(credentials.get("host_url") or "").strip()
    token = str(credentials.get("token") or "").strip()
    if not host_url or not token:
        raise GitBucketReadError(
            "GitBucket is not connected. Run: potpie gitbucket login"
        )
    return host_url, token


def _get_json(
    url: str,
    *,
    headers: dict[str, str],
    http: HttpClient,
) -> Any:
    try:
        response = http.get(url, headers=headers)
    except AuthHttpError as exc:
        raise GitBucketReadError(f"GitBucket request failed: {exc}") from exc
    if response.status_code == 401:
        raise GitBucketReadError(
            "GitBucket authentication failed. Run: potpie gitbucket login"
        )
    if response.status_code not in (200, 201):
        raise GitBucketReadError(
            f"GitBucket API returned HTTP {response.status_code}."
        )
    try:
        return response.json()
    except ValueError as exc:
        raise GitBucketReadError("GitBucket API returned non-JSON.") from exc


def _normalize_repo(repo: dict[str, Any]) -> dict[str, Any]:
    owner_value = repo.get("owner")
    owner_login = (
        str(owner_value.get("login") or "")
        if isinstance(owner_value, dict)
        else ""
    )
    return {
        "full_name": str(repo.get("full_name") or ""),
        "name": str(repo.get("name") or ""),
        "owner": owner_login,
        "private": bool(repo.get("private")),
        "description": str(repo.get("description") or ""),
        "default_branch": str(repo.get("default_branch") or ""),
        "html_url": str(repo.get("html_url") or ""),
        "clone_url": str(repo.get("clone_url") or ""),
    }


def list_gitbucket_repos(
    host_url: str | None = None,
    token: str | None = None,
    *,
    limit: int = _DEFAULT_REPO_LIMIT,
    http: HttpClient | None = None,
) -> list[dict[str, Any]]:
    """Return a list of repos accessible to the authenticated user.

    If ``host_url`` and ``token`` are omitted, loads them from stored credentials.
    """
    host_provided = host_url is not None
    token_provided = token is not None
    if host_provided != token_provided:
        raise GitBucketReadError(
            "GitBucket host_url and token must be provided together."
        )
    if not host_provided and not token_provided:
        host_url, token = _load_credentials()

    api_base = gitbucket_api_base(host_url)
    headers = _token_auth_header(token)
    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        url = f"{api_base}/user/repos?per_page={limit}"
        data = _get_json(url, headers=headers, http=http)
    finally:
        if owns:
            http.close()

    if not isinstance(data, list):
        return []
    return [
        _normalize_repo(repo)
        for repo in data
        if isinstance(repo, dict)
    ]


__all__ = [
    "GitBucketReadError",
    "list_gitbucket_repos",
]
