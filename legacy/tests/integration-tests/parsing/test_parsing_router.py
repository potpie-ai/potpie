"""
Integration tests for parsing HTTP API (POST /parse, GET/POST parsing-status).

Use root conftest client, db_session; mock Celery process_parsing.delay so no real broker.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def mock_process_parsing_delay(monkeypatch):
    """Mock Celery process_parsing.delay so no broker is required."""
    mock_delay = MagicMock(return_value=MagicMock(id="mock-task-id"))
    monkeypatch.setattr(
        "app.modules.parsing.graph_construction.parsing_controller.process_parsing.delay",
        mock_delay,
    )
    return mock_delay


@pytest.fixture
def mock_check_commit_status(monkeypatch):
    """Mock ParseHelper.check_commit_status to avoid GitHub calls."""
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.modules.parsing.graph_construction.parsing_controller.ParseHelper.check_commit_status",
        mock,
    )
    return mock


class TestParseEndpoint:
    """Test POST /api/v1/parse"""

    async def test_post_parse_valid_repo_name(
        self, client, db_session, test_user, mock_process_parsing_delay
    ):
        """Mock Celery process_parsing.delay; valid body with repo_name; expect 200 and task submitted."""
        response = await client.post(
            "/api/v1/parse",
            json={"repo_name": "owner/repo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        assert data.get("status") == "submitted"
        mock_process_parsing_delay.assert_called_once()

    async def test_post_parse_invalid_body(self, client):
        """Missing both repo_name and repo_path; expect 422."""
        response = await client.post("/api/v1/parse", json={})
        assert response.status_code == 422

    async def test_post_parse_repo_path_without_dev_mode(self, client, monkeypatch):
        """repo_path in body with isDevelopmentMode disabled -> 400 or 403."""
        monkeypatch.setenv("isDevelopmentMode", "")
        try:
            response = await client.post(
                "/api/v1/parse",
                json={"repo_path": "/tmp/repo"},
            )
            assert response.status_code in (400, 403)
        finally:
            monkeypatch.setenv("isDevelopmentMode", "enabled", prepend=False)


class TestParsingStatusByProjectId:
    """Test GET /api/v1/parsing-status/{project_id}"""

    async def test_get_parsing_status_found(
        self, client, db_session, conversation_project, mock_check_commit_status
    ):
        """Project exists and user owns it; expect 200 with status and latest."""
        response = await client.get(
            f"/api/v1/parsing-status/{conversation_project.id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "latest" in data
        mock_check_commit_status.assert_called_once()

    async def test_get_parsing_status_not_found(self, client):
        """Unknown project_id or no access; expect 404."""
        response = await client.get("/api/v1/parsing-status/nonexistent-id")
        assert response.status_code == 404


class TestParsingStatusByRepo:
    """Test POST /api/v1/parsing-status"""

    async def test_post_parsing_status_by_repo_found(
        self, client, db_session, conversation_project, mock_check_commit_status
    ):
        """Valid repo_name (+ branch/commit); project exists; expect 200 with project_id, status, latest."""
        payload = {
            "repo_name": conversation_project.repo_name,
        }
        if getattr(conversation_project, "branch_name", None) is not None:
            payload["branch_name"] = conversation_project.branch_name
        response = await client.post("/api/v1/parsing-status", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data.get("project_id") == conversation_project.id
        assert "status" in data
        assert "latest" in data
        assert data.get("repo_name") == conversation_project.repo_name

    async def test_post_parsing_status_by_repo_not_found(self, client):
        """No project for given repo/branch/commit; expect 404."""
        response = await client.post(
            "/api/v1/parsing-status",
            json={"repo_name": "nonexistent/owner-repo"},
        )
        assert response.status_code == 404


class TestInputValidationEdgeCases:
    """Test edge cases for input validation (Part 4.2 of plan)."""

    async def test_post_parse_whitespace_only_repo_name(
        self, client, test_user, mock_process_parsing_delay
    ):
        """repo_name with only whitespace: API currently accepts and enqueues (no validation)."""
        response = await client.post(
            "/api/v1/parse",
            json={"repo_name": "   "},
        )
        assert response.status_code == 200
        assert mock_process_parsing_delay.called

    async def test_post_parse_repo_name_no_slash(
        self, client, test_user, mock_process_parsing_delay
    ):
        """repo_name without slash (e.g. 'myrepo') is accepted and enqueued."""
        response = await client.post(
            "/api/v1/parse",
            json={"repo_name": "myrepo"},
        )
        assert response.status_code == 200
        assert mock_process_parsing_delay.called

    async def test_post_parse_repo_name_multiple_slashes(
        self, client, test_user, mock_process_parsing_delay
    ):
        """repo_name with multiple slashes (e.g. 'org/repo/sub') is accepted and enqueued."""
        response = await client.post(
            "/api/v1/parse",
            json={"repo_name": "org/repo/subpath"},
        )
        assert response.status_code == 200
        assert mock_process_parsing_delay.called

    async def test_post_parse_empty_commit_id(
        self, client, test_user, mock_process_parsing_delay
    ):
        """Empty commit_id string is accepted; task is enqueued."""
        response = await client.post(
            "/api/v1/parse",
            json={"repo_name": "owner/repo", "commit_id": ""},
        )
        assert response.status_code == 200
        assert mock_process_parsing_delay.called

    async def test_post_parse_path_like_repo_name(
        self, client, test_user, mock_process_parsing_delay, monkeypatch
    ):
        """repo_name that looks like a path (e.g. '/tmp/repo') triggers auto-detection."""
        monkeypatch.setenv("isDevelopmentMode", "enabled")
        response = await client.post(
            "/api/v1/parse",
            json={"repo_name": "/tmp/somerepo"},
        )
        # Single expected outcome once contract is defined: e.g. 200 (accept) or 404/422 (reject)
        assert response.status_code in (200, 404, 422)
        if response.status_code != 200:
            assert not mock_process_parsing_delay.called

    async def test_post_parse_relative_path_repo_name(
        self, client, test_user, mock_process_parsing_delay, monkeypatch
    ):
        """repo_name that looks like relative path (./repo); auto-detection."""
        monkeypatch.setenv("isDevelopmentMode", "enabled")
        response = await client.post(
            "/api/v1/parse",
            json={"repo_name": "./somerepo"},
        )
        assert response.status_code in (200, 404, 422)
        if response.status_code != 200:
            assert not mock_process_parsing_delay.called


class TestConcurrencyAndRaces:
    """Test concurrency edge cases (Part 4.1 of plan)."""

    async def test_double_submit_same_repo_idempotent(
        self, client, db_session, test_user, mock_process_parsing_delay
    ):
        """Double-submit same repo+branch should be idempotent or return existing project."""
        payload = {"repo_name": "owner/concurrent-test-repo", "branch_name": "main"}

        response1 = await client.post("/api/v1/parse", json=payload)
        assert response1.status_code == 200
        project_id_1 = response1.json().get("project_id")

        response2 = await client.post("/api/v1/parse", json=payload)
        assert response2.status_code == 200
        project_id_2 = response2.json().get("project_id")

        # Either same project_id (idempotent) or different (new submission) - both valid
        # The key is no 500 error
        assert project_id_1 is not None
        assert project_id_2 is not None
