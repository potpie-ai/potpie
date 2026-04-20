"""HTTP client for Potpie /api/v2/context (X-API-Key)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote

import httpx

CONTEXT_API_PREFIX = "/api/v2/context"


class PotpieContextApiError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail!r}")


def _json_body_for_httpx(obj: Any) -> Any:
    """Recursively convert datetime to ISO strings for JSON bodies."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_body_for_httpx(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_body_for_httpx(x) for x in obj]
    return obj


class PotpieContextApiClient:
    """Thin client: search, ingest, reset, and query/* used by CLI and MCP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key.strip()
        self._timeout = timeout

    def _url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self._base}{CONTEXT_API_PREFIX}{p}"

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, r: httpx.Response) -> None:
        if r.is_success:
            return
        detail: Any
        try:
            detail = r.json()
        except Exception:
            detail = r.text or r.reason_phrase
        raise PotpieContextApiError(r.status_code, detail)

    def list_context_pots(self) -> list[dict[str, Any]]:
        """GET /api/v2/context/pots — user-owned context pots (independent of projects)."""
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(
                self._url("/pots"),
                headers={
                    "X-API-Key": self._api_key,
                    "Accept": "application/json",
                },
            )
        self._raise_for_status(r)
        data = r.json()
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def create_context_pot(
        self,
        *,
        slug: str,
        display_name: Optional[str] = None,
        pot_id: Optional[str] = None,
        primary_repo_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/v2/context/pots."""
        body: dict[str, Any] = {"slug": slug}
        if display_name is not None:
            body["display_name"] = display_name
        if pot_id is not None:
            body["id"] = pot_id
        if primary_repo_name is not None:
            body["primary_repo_name"] = primary_repo_name
        r = self.post_context("/pots", json_body=body)
        self._raise_for_status(r)
        out = r.json()
        return out if isinstance(out, dict) else {}

    def get_context_pot_slug_availability(self, slug: str) -> dict[str, Any]:
        """GET /api/v2/context/pots/slug-availability/{slug}."""
        encoded_slug = quote(slug.strip(), safe="")
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(
                self._url(f"/pots/slug-availability/{encoded_slug}"),
                headers={
                    "X-API-Key": self._api_key,
                    "Accept": "application/json",
                },
            )
        self._raise_for_status(r)
        out = r.json()
        return out if isinstance(out, dict) else {}

    def find_context_pot_by_slug(self, slug: str) -> dict[str, Any] | None:
        """Resolve an accessible pot by slug using the authenticated pot list."""
        wanted = slug.strip().lower()
        if not wanted:
            return None
        for row in self.list_context_pots():
            if str(row.get("slug") or "").strip().lower() == wanted:
                return row
        return None

    def list_pot_repositories(self, pot_id: str) -> list[dict[str, Any]]:
        """GET /api/v2/context/pots/{pot_id}/repositories."""
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(
                self._url(f"/pots/{pot_id}/repositories"),
                headers={
                    "X-API-Key": self._api_key,
                    "Accept": "application/json",
                },
            )
        self._raise_for_status(r)
        data = r.json()
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def add_pot_repository(
        self,
        pot_id: str,
        *,
        owner: str,
        repo: str,
        provider: str = "github",
        provider_host: str = "github.com",
    ) -> dict[str, Any]:
        """POST /api/v2/context/pots/{pot_id}/repositories."""
        body = {
            "owner": owner,
            "repo": repo,
            "provider": provider,
            "provider_host": provider_host,
        }
        r = self.post_context(f"/pots/{pot_id}/repositories", json_body=body)
        self._raise_for_status(r)
        out = r.json()
        return out if isinstance(out, dict) else {}

    def get_health(self) -> tuple[int, Optional[dict[str, Any]]]:
        """GET /health on the same host as base_url."""
        with httpx.Client(timeout=min(self._timeout, 30.0)) as client:
            r = client.get(
                f"{self._base}/health", headers={"Accept": "application/json"}
            )
        if r.status_code != 200:
            return r.status_code, None
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, None

    def post_context(
        self,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        body = _json_body_for_httpx(json_body) if json_body is not None else None
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(
                self._url(path),
                headers=self._headers(),
                json=body,
                params=params,
            )

    def get_context(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        with httpx.Client(timeout=self._timeout) as client:
            return client.get(
                self._url(path),
                headers={
                    "X-API-Key": self._api_key,
                    "Accept": "application/json",
                },
                params=params,
            )

    def search(self, body: dict[str, Any]) -> list[Any]:
        r = self.post_context("/query/search", json_body=body)
        self._raise_for_status(r)
        return r.json()

    def ingest(self, body: dict[str, Any], *, sync: bool) -> tuple[int, dict[str, Any]]:
        params = {"sync": "true"} if sync else None
        r = self.post_context("/ingest", json_body=body, params=params)
        if r.status_code in (200, 202):
            try:
                return r.status_code, r.json()
            except json.JSONDecodeError:
                raise PotpieContextApiError(r.status_code, r.text) from None
        if r.status_code == 409:
            try:
                detail = r.json().get("detail", {})
                if isinstance(detail, dict) and detail.get("error") == "duplicate_ingest":
                    return r.status_code, detail
            except Exception:
                pass
        self._raise_for_status(r)
        return r.status_code, {}

    def get_event(self, event_id: str) -> dict[str, Any]:
        encoded_event_id = quote(event_id.strip(), safe="")
        r = self.get_context(f"/events/{encoded_event_id}")
        self._raise_for_status(r)
        out = r.json()
        return out if isinstance(out, dict) else {}

    def list_events(
        self,
        pot_id: str,
        *,
        limit: int = 20,
        status: Optional[str] = None,
        ingestion_kind: Optional[str] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if ingestion_kind:
            params["ingestion_kind"] = ingestion_kind
        encoded_pot_id = quote(pot_id.strip(), safe="")
        r = self.get_context(f"/pots/{encoded_pot_id}/events", params=params)
        self._raise_for_status(r)
        out = r.json()
        return out if isinstance(out, dict) else {"items": [], "next_cursor": None}

    def reset(self, body: dict[str, Any]) -> dict[str, Any]:
        r = self.post_context("/reset", json_body=body)
        self._raise_for_status(r)
        return r.json()

    def record(self, body: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
        params = {"sync": "true"} if sync else None
        r = self.post_context("/record", json_body=body, params=params)
        self._raise_for_status(r)
        out = r.json()
        return out if isinstance(out, dict) else {}

    def status(self, body: dict[str, Any]) -> dict[str, Any]:
        r = self.post_context("/status", json_body=body)
        self._raise_for_status(r)
        out = r.json()
        return out if isinstance(out, dict) else {}

    def post_query(self, subpath: str, body: dict[str, Any]) -> Any:
        path = f"/query/{subpath.lstrip('/')}"
        r = self.post_context(path, json_body=body)
        self._raise_for_status(r)
        return r.json()

    async def post_query_async(self, subpath: str, body: dict[str, Any]) -> Any:
        path = f"/query/{subpath.lstrip('/')}"
        payload = _json_body_for_httpx(body)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                self._url(path),
                headers=self._headers(),
                json=payload,
            )
        self._raise_for_status(r)
        return r.json()
