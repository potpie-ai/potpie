"""HTTP client for Potpie /api/v2/context (API key or Firebase Bearer)."""

from __future__ import annotations

from collections.abc import Callable
import json
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote

import httpx

from potpie_context_core.domain.errors import CapabilityNotImplemented

CONTEXT_API_PREFIX = "/api/v2/context"


class PotpieContextApiError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail!r}")


class IngestRejectedError(Exception):
    """Server returned HTTP 422 with a structured reconciliation rejection body."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        super().__init__(
            str(body.get("status") or body.get("error") or "reconciliation_rejected")
        )


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
    """Thin client for context graph query, ingest, reset, and support APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        auth_headers: dict[str, str] | None = None,
        auth_headers_provider: Callable[[], dict[str, str]] | None = None,
        reauth_provider: Callable[[], dict[str, str]] | None = None,
        timeout: float = 120.0,
        client_surface: str | None = None,
        client_name: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = (api_key or "").strip()
        self._auth_headers = dict(auth_headers or {})
        self._auth_headers_provider = auth_headers_provider
        # Called only on a 401 to force-refresh auth (e.g. a new Firebase ID
        # token). Distinct from auth_headers_provider, which returns the cached
        # headers used for the normal request.
        self._reauth_provider = reauth_provider
        self._timeout = timeout
        self._client_surface = (client_surface or "").strip() or None
        self._client_name = (client_name or "").strip() or None

    def _url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self._base}{CONTEXT_API_PREFIX}{p}"

    def _client_headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._client_surface:
            h["X-Potpie-Client"] = self._client_surface
        if self._client_name:
            h["X-Potpie-Client-Name"] = self._client_name
        return h

    def _headers(self) -> dict[str, str]:
        auth_headers = (
            self._auth_headers_provider()
            if self._auth_headers_provider is not None
            else dict(self._auth_headers)
        )
        if not auth_headers and self._api_key:
            auth_headers = {"X-API-Key": self._api_key}
        base = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        base.update(auth_headers)
        base.update(self._client_headers())
        return base

    def _get_headers(self) -> dict[str, str]:
        headers = self._headers()
        headers.pop("Content-Type", None)
        return headers

    def _refresh_auth_headers(self) -> bool:
        """Force-refresh auth headers after a 401.

        Returns True only when fresh headers were obtained AND differ from those
        just sent — so the caller retries only when a retry can actually succeed.
        A static API key (unchanged headers) yields False and no wasted retry.
        """
        if self._reauth_provider is None:
            return False
        try:
            fresh = dict(self._reauth_provider() or {})
        except Exception:
            return False
        if not fresh:
            return False
        previous = (
            self._auth_headers_provider()
            if self._auth_headers_provider is not None
            else self._auth_headers
        )
        if fresh == dict(previous or {}):
            return False
        # Pin the refreshed headers so _headers() uses them for the retry.
        self._auth_headers = fresh
        self._auth_headers_provider = None
        return True

    def _raise_for_status(self, r: httpx.Response) -> None:
        if r.is_success:
            return
        detail: Any
        try:
            detail = r.json()
        except Exception:
            detail = r.text or r.reason_phrase
        raise PotpieContextApiError(r.status_code, detail)

    def _get_with_auth_retry(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._get_headers(), params=params)
            if response.status_code == 401 and self._refresh_auth_headers():
                response = client.get(url, headers=self._get_headers(), params=params)
            return response

    def _post_with_auth_retry(
        self,
        url: str,
        *,
        json_body: Any = None,
        params: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                url,
                headers=self._headers(),
                json=json_body,
                params=params,
            )
            if response.status_code == 401 and self._refresh_auth_headers():
                response = client.post(
                    url,
                    headers=self._headers(),
                    json=json_body,
                    params=params,
                )
            return response

    def list_context_pots(self) -> list[dict[str, Any]]:
        """GET /api/v2/context/pots — user-owned context pots (independent of projects)."""
        r = self._get_with_auth_retry(self._url("/pots"))
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
        r = self._get_with_auth_retry(
            self._url(f"/pots/slug-availability/{encoded_slug}")
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
        r = self._get_with_auth_retry(self._url(f"/pots/{pot_id}/repositories"))
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

    def submit_event(
        self,
        *,
        pot_id: str,
        source_system: str,
        event_type: str,
        action: str,
        source_id: str,
        payload: dict[str, Any] | None = None,
        repo_name: str | None = None,
        provider: str | None = "github",
        provider_host: str | None = "github.com",
        event_id: str | None = None,
        ingestion_kind: str | None = None,
        occurred_at: datetime | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """POST /api/v2/context/events/reconcile.

        Used by host-side triggers (CLI, scripts) to drop a normalized context
        event into the ingestion submission pipeline. Returns the raw status +
        body so callers can branch on 202 (queued) vs 409 (duplicate).
        """
        body: dict[str, Any] = {
            "pot_id": pot_id,
            "source_system": source_system,
            "event_type": event_type,
            "action": action,
            "source_id": source_id,
            "payload": payload or {},
        }
        if repo_name is not None:
            body["repo_name"] = repo_name
        if provider is not None:
            body["provider"] = provider
        if provider_host is not None:
            body["provider_host"] = provider_host
        if event_id is not None:
            body["event_id"] = event_id
        if ingestion_kind is not None:
            body["ingestion_kind"] = ingestion_kind
        if occurred_at is not None:
            body["occurred_at"] = occurred_at
        r = self.post_context("/events/reconcile", json_body=body)
        if r.status_code in (200, 202):
            try:
                return r.status_code, r.json()
            except json.JSONDecodeError:
                raise PotpieContextApiError(r.status_code, r.text) from None
        if r.status_code == 409:
            try:
                detail = r.json().get("detail", {})
            except Exception:
                detail = {}
            if isinstance(detail, dict) and (
                detail.get("error") == "duplicate_event" or detail.get("event_id")
            ):
                return r.status_code, detail
        self._raise_for_status(r)
        return r.status_code, {}

    def classify_modified_edges(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /maintenance/classify-modified-edges (dry-run by default)."""
        r = self.post_context("/maintenance/classify-modified-edges", json_body=body)
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
        return self._post_with_auth_retry(
            self._url(path),
            json_body=body,
            params=params,
        )

    def get_context(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        return self._get_with_auth_retry(self._url(path), params=params)

    def context_graph_query(self, body: dict[str, Any]) -> dict[str, Any]:
        del body
        raise CapabilityNotImplemented(
            "http.context_graph_query",
            detail=(
                "remote ContextGraphQuery is no longer a supported client "
                "surface; use the local HostShell/GraphService path."
            ),
            recommended_next_action="Use context_resolve/context_search or graph read locally.",
        )

    def ingest(self, body: dict[str, Any], *, sync: bool) -> tuple[int, dict[str, Any]]:
        params = {"sync": "true"} if sync else None
        r = self.post_context("/ingest", json_body=body, params=params)
        if r.status_code == 422:
            payload: dict[str, Any]
            try:
                raw = r.json()
                payload = raw if isinstance(raw, dict) else {}
            except json.JSONDecodeError:
                payload = {
                    "status": "reconciliation_rejected",
                    "errors": [],
                    "event_id": None,
                    "mutation_id": None,
                    "downgrades": [],
                }
            raise IngestRejectedError(payload)
        if r.status_code in (200, 202):
            try:
                return r.status_code, r.json()
            except json.JSONDecodeError:
                raise PotpieContextApiError(r.status_code, r.text) from None
        if r.status_code == 409:
            try:
                detail = r.json().get("detail", {})
                if (
                    isinstance(detail, dict)
                    and detail.get("error") == "duplicate_ingest"
                ):
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

    async def context_graph_query_async(self, body: dict[str, Any]) -> dict[str, Any]:
        del body
        raise CapabilityNotImplemented(
            "http.context_graph_query_async",
            detail=(
                "remote ContextGraphQuery is no longer a supported client "
                "surface; use the local HostShell/GraphService path."
            ),
            recommended_next_action="Use context_resolve/context_search or graph read locally.",
        )
