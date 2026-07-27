"""GitLab read client: projects, merge requests, and issues.

Data-fetching layer for the GitLab CLI integration. The interactive
picker flow lives in ``adapters.inbound.cli.auth.gitlab_read``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import quote_plus

from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.gitlab_client import (
    gitlab_auth_headers,
    normalize_instance_url,
)
from adapters.outbound.cli_auth.credentials_store import (
    get_gitlab_credentials,
)
from adapters.outbound.cli_auth.provider_config import (
    GITLAB_DEFAULT_INSTANCE,
    gitlab_api_base_url,
)


_HTTP_TIMEOUT = 30.0


class GitLabReadError(Exception):
    """Non-transport failure in a GitLab read operation."""


def load_gitlab_read_credentials(
    instance_host: str | None = None,
) -> dict[str, Any]:
    """Load stored GitLab credentials for API reads."""
    creds = get_gitlab_credentials(instance_host=instance_host)
    if not creds:
        raise GitLabReadError("GitLab is not connected. Run: potpie gitlab login")
    pat = str(creds.get("personal_access_token") or "").strip()
    if not pat:
        raise GitLabReadError(
            "GitLab token not found in local credentials. Run: potpie gitlab login"
        )
    return creds


def _api_base(creds: dict[str, Any]) -> str:
    instance_url = str(creds.get("instance_url") or "").strip()
    normalized = (
        normalize_instance_url(instance_url)
        if instance_url
        else GITLAB_DEFAULT_INSTANCE
    )
    return gitlab_api_base_url(normalized)


def _headers(creds: dict[str, Any]) -> dict[str, str]:
    pat = str(creds.get("personal_access_token") or "").strip()
    return gitlab_auth_headers(pat)


def _get_json(
    url: str,
    headers: dict[str, str],
    *,
    params: dict[str, str] | None = None,
    http: HttpClient | None = None,
) -> Any:
    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        try:
            response = http.get(url, headers=headers, params=params)
        except AuthHttpError as exc:
            raise GitLabReadError(f"GitLab API request failed: {exc}") from exc
        if response.status_code == 401:
            raise GitLabReadError(
                "GitLab token expired or revoked. Run: potpie gitlab login"
            )
        if response.status_code == 403:
            raise GitLabReadError(
                "GitLab token lacks required scopes (need read_api). "
                "Run: potpie gitlab login --force"
            )
        if response.status_code != 200:
            raise GitLabReadError(f"GitLab API returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise GitLabReadError("GitLab API returned non-JSON response") from exc
    finally:
        if owns:
            http.close()


def _encode_project_id(project_id: int | str) -> int | str:
    if isinstance(project_id, str):
        return quote_plus(project_id)
    return project_id


def _fetch_normalized_list(
    path: str,
    *,
    instance_host: str | None,
    params: dict[str, str],
    mapper: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    creds = load_gitlab_read_credentials(instance_host=instance_host)
    data = _get_json(f"{_api_base(creds)}{path}", _headers(creds), params=params)
    if not isinstance(data, list):
        return []
    return [mapper(item) for item in data if isinstance(item, dict)]


def fetch_gitlab_projects(
    *,
    instance_host: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List projects accessible to the authenticated user."""
    return _fetch_normalized_list(
        "/projects",
        instance_host=instance_host,
        params={
            "membership": "true",
            "order_by": "last_activity_at",
            "sort": "desc",
            "per_page": str(min(limit, 100)),
        },
        mapper=lambda item: {
            "id": item.get("id"),
            "path_with_namespace": item.get("path_with_namespace"),
            "name": item.get("name"),
            "visibility": item.get("visibility"),
            "web_url": item.get("web_url"),
            "default_branch": item.get("default_branch"),
        },
    )


def fetch_gitlab_merge_requests(
    project_id: int | str,
    *,
    instance_host: str | None = None,
    state: str = "opened",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch merge requests for a project."""
    encoded_id = _encode_project_id(project_id)
    return _fetch_normalized_list(
        f"/projects/{encoded_id}/merge_requests",
        instance_host=instance_host,
        params={
            "state": state,
            "order_by": "updated_at",
            "sort": "desc",
            "per_page": str(min(limit, 100)),
        },
        mapper=lambda item: {
            "iid": item.get("iid"),
            "title": item.get("title"),
            "state": item.get("state"),
            "author": (
                item.get("author", {}).get("username")
                if isinstance(item.get("author"), dict)
                else None
            ),
            "target_branch": item.get("target_branch"),
            "source_branch": item.get("source_branch"),
            "web_url": item.get("web_url"),
            "updated_at": item.get("updated_at"),
        },
    )


def fetch_gitlab_issues(
    project_id: int | str,
    *,
    instance_host: str | None = None,
    state: str = "opened",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch issues for a project."""
    encoded_id = _encode_project_id(project_id)
    return _fetch_normalized_list(
        f"/projects/{encoded_id}/issues",
        instance_host=instance_host,
        params={
            "state": state,
            "order_by": "updated_at",
            "sort": "desc",
            "per_page": str(min(limit, 100)),
        },
        mapper=lambda item: {
            "iid": item.get("iid"),
            "title": item.get("title"),
            "state": item.get("state"),
            "assignee": (
                item.get("assignee", {}).get("username")
                if isinstance(item.get("assignee"), dict)
                else None
            ),
            "labels": item.get("labels"),
            "web_url": item.get("web_url"),
            "updated_at": item.get("updated_at"),
        },
    )
