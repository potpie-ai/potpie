"""HTTP client for Potpie local-development routes under ``/api/v1``."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

import httpx

LOCAL_API_PREFIX = "/api/v1"


class PotpieLocalApiError(Exception):
    """Raised when the local Potpie API returns an error response."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail!r}")


class PotpieLocalApiClient:
    """Thin client for local Potpie parse/chat development workflows."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 1800.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        token = (bearer_token or "").strip()
        self._bearer_token = token or None
        self._timeout = timeout

    def _url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self._base}{LOCAL_API_PREFIX}{p}"

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        return headers

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail: Any
        try:
            detail = response.json()
        except Exception:
            detail = response.text or response.reason_phrase
        raise PotpieLocalApiError(response.status_code, detail)

    def get_health(self) -> tuple[int, dict[str, Any] | None]:
        with httpx.Client(timeout=min(self._timeout, 30.0)) as client:
            response = client.get(
                f"{self._base}/health", headers={"Accept": "application/json"}
            )
        if response.status_code != 200:
            return response.status_code, None
        try:
            payload = response.json()
        except Exception:
            payload = None
        return response.status_code, payload if isinstance(payload, dict) else None

    def parse_directory(
        self,
        *,
        repo_path: str,
        branch_name: str | None = None,
        repo_name: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"repo_path": repo_path}
        if branch_name:
            body["branch_name"] = branch_name
        if repo_name:
            body["repo_name"] = repo_name
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                self._url("/parse"),
                headers=self._headers(json_body=True),
                json=body,
            )
        self._raise_for_status(response)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def get_parsing_status(self, project_id: str) -> dict[str, Any]:
        encoded = quote(project_id.strip(), safe="")
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                self._url(f"/parsing-status/{encoded}"),
                headers=self._headers(),
            )
        self._raise_for_status(response)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def list_projects(self) -> list[dict[str, Any]]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                self._url("/projects/list"),
                headers=self._headers(),
            )
        self._raise_for_status(response)
        payload = response.json()
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []

    def list_available_agents(self) -> list[dict[str, Any]]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                self._url("/list-available-agents/"),
                headers=self._headers(),
                params={"list_system_agents": "true"},
            )
        self._raise_for_status(response)
        payload = response.json()
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []

    def create_conversation(
        self,
        *,
        project_id: str,
        agent_id: str,
        title: str = "CLI Chat",
        hidden: bool = True,
        user_id: str = "cli",
    ) -> dict[str, Any]:
        body = {
            "user_id": user_id,
            "title": title,
            "status": "active",
            "project_ids": [project_id],
            "agent_ids": [agent_id],
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                self._url("/conversations"),
                headers=self._headers(json_body=True),
                json=body,
                params={"hidden": str(hidden).lower()},
            )
        self._raise_for_status(response)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def send_message(self, conversation_id: str, content: str) -> dict[str, Any]:
        encoded = quote(conversation_id.strip(), safe="")
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                self._url(f"/conversations/{encoded}/message"),
                headers=self._headers(),
                data={"content": content},
                params={"stream": "false"},
            )
        self._raise_for_status(response)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
