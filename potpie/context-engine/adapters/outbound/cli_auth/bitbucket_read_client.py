"""Read helpers for Bitbucket Cloud workspace, repository, and PR selection."""

from __future__ import annotations

from typing import Any

from adapters.outbound.cli_auth.bitbucket_client import (
    _basic_auth_header,
    workspace_slugs_from_list_payload,
)
from adapters.outbound.cli_auth.credentials_store import get_bitbucket_credentials
from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.provider_config import (
    BITBUCKET_API_BASE,
    BITBUCKET_USER_WORKSPACES_PATH,
)


class BitbucketReadError(RuntimeError):
    """Raised when Bitbucket read/list operations fail."""


def load_bitbucket_read_credentials() -> dict[str, Any]:
    credentials = get_bitbucket_credentials()
    if not credentials.get("email") or not credentials.get("api_token"):
        raise BitbucketReadError(
            "Bitbucket is not connected. Run: potpie atlassian login"
        )
    return credentials


def _bitbucket_headers(credentials: dict[str, Any]) -> dict[str, str]:
    return {
        "Authorization": _basic_auth_header(
            str(credentials.get("email") or ""),
            str(credentials.get("api_token") or ""),
        ),
        "Accept": "application/json",
    }


def _get_json(
    path: str,
    *,
    credentials: dict[str, Any],
    params: dict[str, Any] | None = None,
    http: HttpClient | None = None,
) -> dict[str, Any]:
    owns = http is None
    http = http or AuthHttpClient(timeout=15.0)
    try:
        response = http.get(
            f"{BITBUCKET_API_BASE}{path}",
            headers=_bitbucket_headers(credentials),
            params=params or {},
        )
    except AuthHttpError as exc:
        raise BitbucketReadError("Bitbucket request failed.") from exc
    finally:
        if owns:
            http.close()
    if response.status_code == 401:
        raise BitbucketReadError(
            "Bitbucket authentication failed. Run: potpie atlassian login"
        )
    if response.status_code == 403:
        raise BitbucketReadError(
            "Bitbucket API token missing required read scopes."
        )
    if response.status_code != 200:
        raise BitbucketReadError(
            f"Bitbucket request failed with status {response.status_code}."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise BitbucketReadError("Bitbucket returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise BitbucketReadError("Bitbucket returned an unexpected response.")
    return payload


def fetch_bitbucket_workspaces(
    *,
    limit: int = 50,
    credentials: dict[str, Any] | None = None,
    http: HttpClient | None = None,
) -> list[dict[str, Any]]:
    creds = credentials or load_bitbucket_read_credentials()
    payload = _get_json(
        BITBUCKET_USER_WORKSPACES_PATH,
        credentials=creds,
        params={"pagelen": max(1, min(limit, 100))},
        http=http,
    )
    rows: list[dict[str, Any]] = []
    for row in payload.get("values") or []:
        if not isinstance(row, dict):
            continue
        workspace = row.get("workspace")
        record = workspace if isinstance(workspace, dict) else row
        key = str(record.get("slug") or record.get("username") or "").strip()
        if not key:
            continue
        rows.append(
            {
                "id": str(record.get("uuid") or "").strip(),
                "key": key,
                "name": str(record.get("name") or key).strip(),
                "type": "workspace",
            }
        )
    if not rows:
        for key in workspace_slugs_from_list_payload(payload):
            rows.append({"id": "", "key": key, "name": key, "type": "workspace"})
    return rows


def fetch_bitbucket_repositories(
    workspace_key: str,
    *,
    limit: int = 50,
    credentials: dict[str, Any] | None = None,
    http: HttpClient | None = None,
) -> list[dict[str, Any]]:
    workspace = workspace_key.strip()
    if not workspace:
        raise BitbucketReadError("Bitbucket workspace key is required.")
    creds = credentials or load_bitbucket_read_credentials()
    payload = _get_json(
        f"/repositories/{workspace}",
        credentials=creds,
        params={"pagelen": max(1, min(limit, 100)), "sort": "-updated_on"},
        http=http,
    )
    rows: list[dict[str, Any]] = []
    for row in payload.get("values") or []:
        if not isinstance(row, dict):
            continue
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue
        links = row.get("links") if isinstance(row.get("links"), dict) else {}
        html = links.get("html") if isinstance(links.get("html"), dict) else {}
        rows.append(
            {
                "id": str(row.get("uuid") or "").strip(),
                "key": slug,
                "name": str(row.get("name") or slug).strip(),
                "type": "repository",
                "url": str(html.get("href") or "").strip(),
            }
        )
    return rows


def fetch_bitbucket_pull_requests(
    workspace_key: str,
    repo_slug: str,
    *,
    limit: int = 10,
    credentials: dict[str, Any] | None = None,
    http: HttpClient | None = None,
) -> list[dict[str, Any]]:
    workspace = workspace_key.strip()
    repo = repo_slug.strip()
    if not workspace or not repo:
        raise BitbucketReadError("Bitbucket workspace and repository are required.")
    creds = credentials or load_bitbucket_read_credentials()
    payload = _get_json(
        f"/repositories/{workspace}/{repo}/pullrequests",
        credentials=creds,
        params={"pagelen": max(1, min(limit, 50)), "sort": "-updated_on"},
        http=http,
    )
    rows: list[dict[str, Any]] = []
    for row in payload.get("values") or []:
        if not isinstance(row, dict):
            continue
        author = row.get("author") if isinstance(row.get("author"), dict) else {}
        links = row.get("links") if isinstance(row.get("links"), dict) else {}
        html = links.get("html") if isinstance(links.get("html"), dict) else {}
        rows.append(
            {
                "id": row.get("id"),
                "title": str(row.get("title") or "").strip(),
                "state": str(row.get("state") or "").strip(),
                "author": str(author.get("display_name") or author.get("nickname") or "").strip(),
                "updated": str(row.get("updated_on") or "").strip(),
                "url": str(html.get("href") or "").strip(),
            }
        )
    return rows
