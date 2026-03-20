"""HTTP client for communicating with the local Potpie server.

Security note: This client is designed exclusively for local development use.
It communicates with a Potpie server running on the same machine via HTTP.
For production deployments, use the hosted Potpie service at https://potpie.ai.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

# Local-only default; override via POTPIE_BASE_URL env var.
# HTTP is intentional here — this CLI targets localhost for development.
DEFAULT_BASE_URL = os.getenv("POTPIE_BASE_URL", "http://localhost:8001")  # noqa: S5332
DEFAULT_TIMEOUT = 30
PARSE_POLL_INTERVAL = 5  # seconds between status polls
PARSE_READY_STATUSES = {"ready"}
PARSE_FAILED_STATUSES = {"error", "failed"}


class PotpieClientError(Exception):
    """Raised when the Potpie server returns an error."""


class PotpieClient:
    """Thin HTTP client for the local Potpie REST API (development mode)."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1/{path.lstrip('/')}"

    def _check(self, response: requests.Response) -> Dict[str, Any]:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise PotpieClientError(f"HTTP {response.status_code}: {detail}") from exc
        return response.json()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(
        self,
        repo_path: Optional[str] = None,
        repo_name: Optional[str] = None,
        branch_name: str = "main",
    ) -> Dict[str, Any]:
        """Submit a repository for parsing.

        Returns the initial response with project_id and status.
        """
        payload: Dict[str, Any] = {"branch_name": branch_name}
        if repo_path:
            payload["repo_path"] = repo_path
        if repo_name:
            payload["repo_name"] = repo_name
        response = self._session.post(
            self._url("parse"), json=payload, timeout=DEFAULT_TIMEOUT
        )
        return self._check(response)

    def get_parsing_status(self, project_id: str) -> Dict[str, Any]:
        """Return the current parsing status for a project."""
        response = self._session.get(
            self._url(f"parsing-status/{project_id}"), timeout=DEFAULT_TIMEOUT
        )
        return self._check(response)

    def poll_parsing_status(
        self,
        project_id: str,
        poll_interval: int = PARSE_POLL_INTERVAL,
    ) -> Iterator[Dict[str, Any]]:
        """Yield status dicts until parsing is complete or fails.

        Each yielded dict has at least ``status`` and ``latest`` keys.
        """
        while True:
            status_data = self.get_parsing_status(project_id)
            yield status_data
            status = (status_data.get("status") or "").lower()
            if status in PARSE_READY_STATUSES or status in PARSE_FAILED_STATUSES:
                return
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def list_projects(self) -> List[Dict[str, Any]]:
        """Return all projects for the current user."""
        response = self._session.get(
            self._url("projects/list"), timeout=DEFAULT_TIMEOUT
        )
        result = self._check(response)
        if isinstance(result, list):
            return result
        # Some endpoints wrap in a dict
        return result.get("projects", [result])

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def list_agents(self, list_system_agents: bool = True) -> List[Dict[str, Any]]:
        """Return available agents."""
        response = self._session.get(
            self._url("list-available-agents/"),
            params={"list_system_agents": list_system_agents},
            timeout=DEFAULT_TIMEOUT,
        )
        result = self._check(response)
        if isinstance(result, list):
            return result
        return result.get("agents", [result])

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        project_id: str,
        agent_id: str,
        user_id: str = "defaultUser",
    ) -> Dict[str, Any]:
        """Create a new conversation and return the response."""
        payload = {
            "user_id": user_id,
            "title": f"CLI chat – {project_id}",
            "status": "active",
            "project_ids": [project_id],
            "agent_ids": [agent_id],
        }
        response = self._session.post(
            self._url("conversations"),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        return self._check(response)

    def send_message(
        self,
        conversation_id: str,
        content: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a message in a conversation and return the response.

        Uses stream=False for simple blocking responses.
        """
        response = self._session.post(
            self._url(f"conversations/{conversation_id}/message"),
            json={"content": content},
            params={"stream": "false" if not stream else "true"},
            timeout=120,
        )
        return self._check(response)

    def health(self) -> bool:
        """Return True if the server is reachable."""
        try:
            resp = self._session.get(
                f"{self.base_url}/health", timeout=5
            )
            return resp.status_code < 500
        except requests.ConnectionError:
            return False
