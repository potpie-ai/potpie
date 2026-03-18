"""HTTP client for communicating with the Potpie API server."""

from __future__ import annotations

import time
from typing import Any, Generator

import httpx

DEFAULT_BASE_URL = "http://localhost:8001"
DEFAULT_TIMEOUT = 30.0


class PotpieClient:
    """Thin HTTP client wrapping the Potpie REST API.

    All methods raise ``httpx.HTTPStatusError`` on non-2xx responses
    so callers can handle errors uniformly.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Check if the Potpie server is running."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def is_alive(self) -> bool:
        """Return True if the server responds to /health."""
        try:
            self.health()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(
        self,
        repo_path: str,
        branch_name: str = "main",
    ) -> dict[str, Any]:
        """Submit a repository for parsing.

        Args:
            repo_path: Local path or remote URL of the repository.
            branch_name: Git branch to parse.

        Returns:
            API response dict containing the project_id.
        """
        resp = self._client.post(
            "/api/v1/parse",
            json={
                "repo_path": repo_path,
                "branch_name": branch_name,
            },
            headers={"isDevelopmentMode": "enabled"},
        )
        resp.raise_for_status()
        return resp.json()

    def get_parsing_status(self, project_id: str) -> dict[str, Any]:
        """Get parsing status for a project.

        Args:
            project_id: The project identifier returned by parse().

        Returns:
            Dict with at least a ``status`` key.
        """
        resp = self._client.get(
            f"/api/v1/parsing-status/{project_id}",
            headers={"isDevelopmentMode": "enabled"},
        )
        resp.raise_for_status()
        return resp.json()

    def poll_parsing(
        self,
        project_id: str,
        interval: float = 3.0,
        max_wait: float = 600.0,
    ) -> dict[str, Any]:
        """Poll parsing status until completion or timeout.

        Args:
            project_id: Project to poll.
            interval: Seconds between polls.
            max_wait: Maximum seconds to wait.

        Returns:
            Final status dict.

        Raises:
            TimeoutError: If parsing does not complete within max_wait.
        """
        start = time.monotonic()
        while True:
            status = self.get_parsing_status(project_id)
            state = status.get("status", "").lower()
            if state in ("ready", "completed", "parsed"):
                return status
            if state in ("error", "failed"):
                return status
            if time.monotonic() - start > max_wait:
                raise TimeoutError(
                    f"Parsing did not complete within {max_wait}s (last status: {state})"
                )
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects."""
        resp = self._client.get(
            "/api/v1/projects/list",
            headers={"isDevelopmentMode": "enabled"},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def list_agents(self) -> list[dict[str, Any]]:
        """List available agents."""
        resp = self._client.get(
            "/api/v1/list-available-agents",
            headers={"isDevelopmentMode": "enabled"},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        project_id: str,
        agent_id: str,
        title: str = "CLI Chat",
    ) -> dict[str, Any]:
        """Create a new conversation.

        Args:
            project_id: The parsed project to converse about.
            agent_id: Agent identifier (e.g. ``"qna_agent"``).
            title: Human-readable conversation title.

        Returns:
            Dict containing the ``conversation_id``.
        """
        resp = self._client.post(
            "/api/v1/conversations/",
            json={
                "project_ids": [project_id],
                "agent_id": agent_id,
                "title": title,
            },
            headers={"isDevelopmentMode": "enabled"},
        )
        resp.raise_for_status()
        return resp.json()

    def send_message(
        self,
        conversation_id: str,
        content: str,
    ) -> Generator[str, None, None]:
        """Send a message and stream the response.

        Args:
            conversation_id: Active conversation ID.
            content: User message text.

        Yields:
            Response text chunks as they arrive.
        """
        with self._client.stream(
            "POST",
            f"/api/v1/conversations/{conversation_id}/message/",
            json={"content": content, "node_ids": []},
            headers={"isDevelopmentMode": "enabled"},
            timeout=300.0,
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_text():
                if chunk:
                    yield chunk
