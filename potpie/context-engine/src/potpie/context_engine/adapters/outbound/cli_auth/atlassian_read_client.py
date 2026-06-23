"""Read-only Atlassian Cloud HTTP/data client (classic unscoped API tokens).

Pure transport + parsing for Jira & Confluence reads (fetch projects/issues/
spaces/pages, credential loading, response parsing) — no CLI/presentation. The
interactive workspace-selection flow lives in
``potpie.context_engine.adapters.inbound.cli.auth.atlassian_read``.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import quote

from potpie.context_engine.adapters.outbound.cli_auth.atlassian_client import (
    AtlassianProduct,
    atlassian_basic_auth_header,
    atlassian_bearer_auth_header,
    normalize_site_url,
)
from potpie.context_engine.adapters.outbound.cli_auth.credentials_store import (
    ProviderCredentialError,
    get_confluence_credentials,
    get_jira_credentials,
)
from potpie.context_engine.adapters.outbound.cli_auth.integration_profile import (
    atlassian_site_from_entry,
    atlassian_workspaces_from_entry,
)
from potpie.context_engine.adapters.outbound.cli_auth.errors import CliAuthError
from potpie.context_engine.adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from potpie.context_engine.adapters.outbound.cli_auth.provider_config import (
    atlassian_confluence_gateway_url,
    atlassian_jira_gateway_url,
)

_HTTP_TIMEOUT = 30.0
_DEFAULT_JIRA_LIMIT = 10
_DEFAULT_CONFLUENCE_LIMIT = 10
_EXCERPT_LEN = 280


class AtlassianReadError(CliAuthError):
    """Failed to read Jira or Confluence data with stored credentials."""


def _is_gateway_base(base: str) -> bool:
    return "api.atlassian.com" in base.lower()


def _auth_header_variants(
    email: str,
    api_token: str,
    *,
    base: str,
) -> list[dict[str, str]]:
    """Build auth headers for a request base URL.

    Tenant URLs (*.atlassian.net) require Basic email:token. Bearer is interpreted
    as a Connect JWT and returns 403 "Failed to parse Connect Session Auth Token".
    """
    accept = {"Accept": "application/json"}
    basic = {**accept, "Authorization": atlassian_basic_auth_header(email, api_token)}
    if _is_gateway_base(base):
        bearer = {**accept, "Authorization": atlassian_bearer_auth_header(api_token)}
        return [basic, bearer]
    return [basic]


def _ordered_bases(
    product: AtlassianProduct,
    cloud_id: str,
    site_url: str,
    *,
    site_first: bool = False,
) -> list[str]:
    gateway = _gateway_bases(product, cloud_id)
    site = _site_bases(product, site_url)
    if site_first:
        return site + gateway
    return gateway + site


def _cloud_id_from_credentials(credentials: dict[str, Any]) -> str:
    site = atlassian_site_from_entry(credentials)
    cloud_id = str(site.get("cloud_id") or credentials.get("cloud_id") or "").strip()
    if not cloud_id:
        raise AtlassianReadError(
            "Missing cloud_id. Run: potpie jira login or potpie confluence login"
        )
    return cloud_id


def _site_url_from_credentials(credentials: dict[str, Any]) -> str:
    site = atlassian_site_from_entry(credentials)
    site_url = normalize_site_url(
        str(site.get("site_url") or credentials.get("site_url") or "")
    )
    if not site_url:
        raise AtlassianReadError(
            "Missing site_url. Run: potpie jira login or potpie confluence login"
        )
    return site_url


def _gateway_bases(product: AtlassianProduct, cloud_id: str) -> list[str]:
    if product == "jira":
        return [atlassian_jira_gateway_url(cloud_id).rstrip("/")]
    return [atlassian_confluence_gateway_url(cloud_id).rstrip("/")]


def _site_bases(product: AtlassianProduct, site_url: str) -> list[str]:
    base = normalize_site_url(site_url)
    if not base:
        return []
    if product == "jira":
        return [base]
    return [base]


def _transport_read_error(
    exc: AuthHttpError,
    *,
    product: AtlassianProduct,
    method: str,
    base: str,
    path: str,
) -> AtlassianReadError:
    return AtlassianReadError(f"{product} {method} failed for {path} via {base}: {exc}")


def _get_json(
    *,
    email: str,
    api_token: str,
    product: AtlassianProduct,
    cloud_id: str,
    site_url: str,
    path: str,
    site_first: bool = False,
    http: HttpClient | None = None,
) -> dict[str, Any]:
    path = path if path.startswith("/") else f"/{path}"
    paths = [path]
    if product == "confluence" and not path.startswith("/wiki/"):
        paths.append(f"/wiki{path}")
    bases = _ordered_bases(product, cloud_id, site_url, site_first=site_first)
    last_status: int | None = None
    last_body = ""
    last_transport_exc: AuthHttpError | None = None
    last_transport_base = ""
    last_transport_path = ""

    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        for candidate_path in paths:
            for base in bases:
                url = f"{base}{candidate_path}"
                for headers in _auth_header_variants(email, api_token, base=base):
                    try:
                        response = http.get(url, headers=headers)
                    except AuthHttpError as exc:
                        last_transport_exc = exc
                        last_transport_base = base
                        last_transport_path = candidate_path
                        continue
                    last_status = response.status_code
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, dict):
                            return data
                        return {"data": data}
                    last_body = response.text[:500]
    finally:
        if owns:
            http.close()

    if last_transport_exc is not None and last_status is None:
        raise _transport_read_error(
            last_transport_exc,
            product=product,
            method="GET",
            base=last_transport_base,
            path=last_transport_path,
        ) from last_transport_exc

    raise AtlassianReadError(
        f"{product} read failed (HTTP {last_status}): {last_body or 'no response body'}"
    )


def _post_json(
    *,
    email: str,
    api_token: str,
    product: AtlassianProduct,
    cloud_id: str,
    site_url: str,
    path: str,
    body: dict[str, Any],
    site_first: bool = False,
    http: HttpClient | None = None,
) -> dict[str, Any]:
    path = path if path.startswith("/") else f"/{path}"
    bases = _ordered_bases(product, cloud_id, site_url, site_first=site_first)
    last_status: int | None = None
    last_body = ""
    headers_base = {"Accept": "application/json", "Content-Type": "application/json"}

    owns = http is None
    http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
    try:
        for base in bases:
            url = f"{base}{path}"
            for auth in _auth_header_variants(email, api_token, base=base):
                headers = {**headers_base, **auth}
                try:
                    response = http.post(url, headers=headers, json=body)
                except AuthHttpError as exc:
                    raise _transport_read_error(
                        exc,
                        product=product,
                        method="POST",
                        base=base,
                        path=path,
                    ) from exc
                last_status = response.status_code
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        return data
                    return {"data": data}
                last_body = response.text[:500]
    finally:
        if owns:
            http.close()

    raise AtlassianReadError(
        f"{product} read failed (HTTP {last_status}): {last_body or 'no response body'}"
    )


_JIRA_ISSUE_FIELDS = [
    "summary",
    "status",
    "created",
    "updated",
    "project",
    "assignee",
    "reporter",
    "priority",
    "issuetype",
    "description",
]


def _jira_search(
    ctx: dict[str, Any],
    *,
    jql: str,
    max_results: int,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Run Jira issue search preferring tenant Basic auth and GET /search/jql."""
    field_list = fields or _JIRA_ISSUE_FIELDS
    fields_param = ",".join(field_list)
    encoded_jql = quote(jql, safe="")
    get_path = (
        f"/rest/api/3/search/jql?jql={encoded_jql}"
        f"&maxResults={max_results}&fields={fields_param}"
    )
    try:
        return _get_json(
            email=ctx["email"],
            api_token=ctx["api_token"],
            product="jira",
            cloud_id=ctx["cloud_id"],
            site_url=ctx["site_url"],
            path=get_path,
            site_first=True,
        )
    except AtlassianReadError:
        return _post_json(
            email=ctx["email"],
            api_token=ctx["api_token"],
            product="jira",
            cloud_id=ctx["cloud_id"],
            site_url=ctx["site_url"],
            path="/rest/api/3/search",
            body={
                "jql": jql,
                "maxResults": max_results,
                "fields": field_list,
            },
            site_first=True,
        )


def _normalize_jira_datetime(value: Any) -> str:
    """Return a compact date/time string from Jira ISO timestamps."""
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        date_part, time_part = text.split("T", 1)
        time_part = time_part.rstrip("Zz")
        for offset_marker in ("+", "-"):
            if offset_marker in time_part:
                time_part = time_part.split(offset_marker, 1)[0]
                break
        time_part = time_part.split(".", 1)[0]
        if time_part:
            return f"{date_part} {time_part}"
        return date_part
    return text


def _parse_jira_issues(
    payload: dict[str, Any],
    *,
    site_url: str,
    excerpt_description: bool = True,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for item in payload.get("issues") or []:
        if not isinstance(item, dict):
            continue
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        status = fields.get("status") if isinstance(fields.get("status"), dict) else {}
        project = (
            fields.get("project") if isinstance(fields.get("project"), dict) else {}
        )
        assignee = (
            fields.get("assignee") if isinstance(fields.get("assignee"), dict) else {}
        )
        reporter = (
            fields.get("reporter") if isinstance(fields.get("reporter"), dict) else {}
        )
        priority = (
            fields.get("priority") if isinstance(fields.get("priority"), dict) else {}
        )
        issue_type = (
            fields.get("issuetype") if isinstance(fields.get("issuetype"), dict) else {}
        )
        description = fields.get("description")
        if isinstance(description, dict):
            description_text = _adf_to_plain(description)
        else:
            description_text = str(description or "")
        issue_key = str(item.get("key") or "")
        row: dict[str, Any] = {
            "key": issue_key,
            "summary": fields.get("summary"),
            "status": status.get("name"),
            "project": project.get("key") or project.get("name"),
            "created": _normalize_jira_datetime(fields.get("created")),
            "updated": _normalize_jira_datetime(fields.get("updated")),
            "assignee": assignee.get("displayName") or assignee.get("emailAddress"),
            "reporter": reporter.get("displayName") or reporter.get("emailAddress"),
            "priority": priority.get("name"),
            "type": issue_type.get("name"),
            "url": _issue_browse_url(site_url, issue_key) if issue_key else None,
        }
        if excerpt_description:
            row["description"] = _excerpt(description_text)
        issues.append(row)
    return issues


def _excerpt(text: str, *, max_len: int = _EXCERPT_LEN) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3].rstrip() + "..."


def _adf_to_plain(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return str(node.get("text") or "")
    parts: list[str] = []
    for child in node.get("content") or []:
        parts.append(_adf_to_plain(child))
    return "".join(parts)


def _html_to_plain(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(html or ""))
    return _excerpt(unescape(text))


def _normalize_confluence_date(value: Any) -> str:
    """Return a human-readable date from Confluence history/version fields."""
    if isinstance(value, dict):
        return str(value.get("friendlyWhen") or value.get("when") or "").strip()
    return str(value or "").strip()


def _confluence_author_name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    by = value.get("by")
    if not isinstance(by, dict):
        return ""
    return str(by.get("displayName") or by.get("publicName") or "").strip()


def _parse_confluence_timestamps(
    history: dict[str, Any],
    version: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Return updated, updated_by, created, created_by from Confluence metadata."""
    updated = _normalize_confluence_date(history.get("lastUpdated"))
    updated_by = _confluence_author_name(history.get("lastUpdated"))
    if not updated:
        updated = _normalize_confluence_date(version.get("when"))

    created = _normalize_confluence_date(history.get("createdDate"))
    created_by = _confluence_person_name(history.get("createdBy"))
    return updated, updated_by, created, created_by


def _confluence_person_name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("displayName") or value.get("publicName") or "").strip()


def _issue_browse_url(site_url: str, issue_key: str) -> str:
    base = normalize_site_url(site_url).rstrip("/")
    return f"{base}/browse/{issue_key}"


def _jira_project_url(site_url: str, project_key: str) -> str:
    base = normalize_site_url(site_url).rstrip("/")
    key = str(project_key or "").strip()
    return f"{base}/browse/{key}" if key else base


def _confluence_page_url(site_url: str, webui: str | None) -> str:
    base = normalize_site_url(site_url).rstrip("/")
    path = str(webui or "").strip()
    if not path:
        return base
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    if path.startswith("/wiki/"):
        return f"{base}{path}"
    return f"{base}/wiki{path}"


def _load_product_read_credentials(product: AtlassianProduct) -> dict[str, Any]:
    label = product.capitalize()
    login_cmd = f"potpie {product} login"
    try:
        credentials = (
            get_jira_credentials()
            if product == "jira"
            else get_confluence_credentials()
        )
    except ProviderCredentialError as exc:
        raise AtlassianReadError(str(exc)) from exc
    if not credentials:
        raise AtlassianReadError(f"{label} is not connected. Run: {login_cmd}")
    email = str(credentials.get("email") or "").strip()
    api_token = str(credentials.get("api_token") or "").strip()
    if not email or not api_token:
        raise AtlassianReadError(f"{label} API token missing. Run: {login_cmd}")
    workspaces = atlassian_workspaces_from_entry(credentials)
    return {
        "email": email,
        "api_token": api_token,
        "cloud_id": _cloud_id_from_credentials(credentials),
        "site_url": _site_url_from_credentials(credentials),
        "site_name": str(
            atlassian_site_from_entry(credentials).get("site_name")
            or credentials.get("site_name")
            or ""
        ).strip(),
        "workspaces": workspaces,
    }


def load_jira_read_credentials() -> dict[str, Any]:
    """Return Jira API credentials from stored integration metadata."""
    return _load_product_read_credentials("jira")


def load_confluence_read_credentials() -> dict[str, Any]:
    """Return Confluence API credentials from stored integration metadata."""
    return _load_product_read_credentials("confluence")


def load_atlassian_read_credentials() -> dict[str, Any]:
    """Backwards-compatible alias; prefers Jira credentials."""
    try:
        return load_jira_read_credentials()
    except AtlassianReadError:
        return load_confluence_read_credentials()


def fetch_jira_projects(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return Jira projects (workspaces) for the connected site."""
    ctx = load_jira_read_credentials()
    max_results = max(1, min(int(limit), 50))
    payload = _get_json(
        email=ctx["email"],
        api_token=ctx["api_token"],
        product="jira",
        cloud_id=ctx["cloud_id"],
        site_url=ctx["site_url"],
        path=f"/rest/api/3/project/search?maxResults={max_results}",
    )
    projects: list[dict[str, Any]] = []
    for item in payload.get("values") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        projects.append(
            {
                "key": key,
                "name": item.get("name"),
                "type": item.get("projectTypeKey") or item.get("style"),
                "lead": (item.get("lead") or {}).get("displayName")
                if isinstance(item.get("lead"), dict)
                else None,
                "url": _jira_project_url(ctx["site_url"], key),
            }
        )
    return projects


def fetch_jira_issues_in_project(
    project_key: str,
    *,
    limit: int = _DEFAULT_JIRA_LIMIT,
) -> list[dict[str, Any]]:
    """Return recent Jira issues in a project with summary and description excerpts."""
    ctx = load_jira_read_credentials()
    key = str(project_key or "").strip().upper()
    if not key:
        raise AtlassianReadError("Jira project key is required.")
    max_results = max(1, min(int(limit), 50))
    payload = _jira_search(
        ctx,
        jql=f'project = "{key}" ORDER BY updated DESC',
        max_results=max_results,
    )
    return _parse_jira_issues(payload, site_url=ctx["site_url"])


def fetch_confluence_pages_in_space(
    space_key: str,
    *,
    limit: int = _DEFAULT_CONFLUENCE_LIMIT,
) -> list[dict[str, Any]]:
    """Return Confluence pages in a space with title and body excerpt."""
    ctx = load_confluence_read_credentials()
    key = str(space_key or "").strip().upper()
    if not key:
        raise AtlassianReadError("Confluence space key is required.")
    max_results = max(1, min(int(limit), 50))
    payload = _get_json(
        email=ctx["email"],
        api_token=ctx["api_token"],
        product="confluence",
        cloud_id=ctx["cloud_id"],
        site_url=ctx["site_url"],
        path=(
            f"/wiki/rest/api/content?spaceKey={key}&type=page"
            f"&limit={max_results}"
            f"&expand=body.storage,version,history.lastUpdated,history.createdDate,history.createdBy"
        ),
        site_first=True,
    )
    pages: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        body = item.get("body") if isinstance(item.get("body"), dict) else {}
        storage = body.get("storage") if isinstance(body.get("storage"), dict) else {}
        history = item.get("history") if isinstance(item.get("history"), dict) else {}
        version = item.get("version") if isinstance(item.get("version"), dict) else {}
        links = item.get("_links") if isinstance(item.get("_links"), dict) else {}
        title = str(item.get("title") or "").strip()
        updated, updated_by, created, created_by = _parse_confluence_timestamps(
            history,
            version,
        )
        pages.append(
            {
                "id": item.get("id"),
                "title": title,
                "space": key,
                "status": item.get("status"),
                "updated": updated,
                "updated_by": updated_by,
                "created": created,
                "created_by": created_by,
                "excerpt": _html_to_plain(str(storage.get("value") or "")),
                "url": _confluence_page_url(ctx["site_url"], links.get("webui")),
            }
        )
    return pages


def fetch_jira_issues_sample(
    *, limit: int = _DEFAULT_JIRA_LIMIT
) -> list[dict[str, Any]]:
    """Return recent Jira issues using saved project or the first available project."""
    ctx = load_jira_read_credentials()
    prefs = ctx.get("workspaces") if isinstance(ctx.get("workspaces"), dict) else {}
    project_key = str(prefs.get("jira_project") or "").strip().upper()
    if not project_key:
        projects = fetch_jira_projects(limit=1)
        if projects:
            project_key = str(projects[0].get("key") or "").strip().upper()
    if project_key:
        return fetch_jira_issues_in_project(project_key, limit=limit)
    max_results = max(1, min(int(limit), 50))
    payload = _jira_search(
        ctx,
        jql="order by updated DESC",
        max_results=max_results,
        fields=["summary", "status", "updated", "project"],
    )
    return _parse_jira_issues(
        payload, site_url=ctx["site_url"], excerpt_description=False
    )


def fetch_confluence_content_sample(
    *, limit: int = _DEFAULT_CONFLUENCE_LIMIT
) -> list[dict[str, Any]]:
    """Return pages in saved Confluence space, or spaces if none is saved yet."""
    ctx = load_confluence_read_credentials()
    prefs = ctx.get("workspaces") if isinstance(ctx.get("workspaces"), dict) else {}
    space_key = str(prefs.get("confluence_space") or "").strip().upper()
    if space_key:
        return fetch_confluence_pages_in_space(space_key, limit=limit)
    return fetch_confluence_spaces_sample(limit=limit)


def fetch_confluence_spaces_sample(
    *, limit: int = _DEFAULT_CONFLUENCE_LIMIT
) -> list[dict[str, Any]]:
    """Return Confluence spaces (read-only) using the stored API token."""
    ctx = load_confluence_read_credentials()
    max_results = max(1, min(int(limit), 50))
    payload = _get_json(
        email=ctx["email"],
        api_token=ctx["api_token"],
        product="confluence",
        cloud_id=ctx["cloud_id"],
        site_url=ctx["site_url"],
        path=f"/wiki/rest/api/space?limit={max_results}",
    )
    spaces: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        links = item.get("_links") if isinstance(item.get("_links"), dict) else {}
        spaces.append(
            {
                "key": item.get("key"),
                "name": item.get("name"),
                "type": item.get("type"),
                "webui": links.get("webui"),
                "url": _confluence_page_url(ctx["site_url"], links.get("webui")),
            }
        )
    return spaces
