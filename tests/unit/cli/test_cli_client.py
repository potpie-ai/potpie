"""Unit tests for potpie.cli.client (PotpieClient)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from potpie.cli.client import (
    PARSE_POLL_INTERVAL,
    PotpieClient,
    PotpieClientError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# PotpieClient initialisation
# ---------------------------------------------------------------------------


class TestPotpieClientInit:
    def test_default_base_url(self):
        client = PotpieClient()
        assert client.base_url == "http://localhost:8001"

    def test_custom_base_url(self):
        client = PotpieClient("http://localhost:9999")
        assert client.base_url == "http://localhost:9999"

    def test_trailing_slash_stripped(self):
        client = PotpieClient("http://localhost:8001/")
        assert client.base_url == "http://localhost:8001"

    def test_url_helper(self):
        client = PotpieClient("http://localhost:8001")
        assert client._url("parse") == "http://localhost:8001/api/v1/parse"
        assert client._url("/parse") == "http://localhost:8001/api/v1/parse"


# ---------------------------------------------------------------------------
# parse()
# ---------------------------------------------------------------------------


class TestPotpieClientParse:
    def test_parse_with_repo_path(self):
        client = PotpieClient()
        mock_resp = _mock_response({"project_id": "proj-1", "status": "submitted"})
        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            result = client.parse(repo_path="/tmp/myrepo", branch_name="main")

        assert result["project_id"] == "proj-1"
        assert result["status"] == "submitted"
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["repo_path"] == "/tmp/myrepo"
        assert kwargs["json"]["branch_name"] == "main"

    def test_parse_with_repo_name(self):
        client = PotpieClient()
        mock_resp = _mock_response({"project_id": "proj-2", "status": "submitted"})
        with patch.object(client._session, "post", return_value=mock_resp):
            result = client.parse(repo_name="owner/repo", branch_name="dev")

        assert result["project_id"] == "proj-2"

    def test_parse_raises_on_error_status(self):
        client = PotpieClient()
        mock_resp = _mock_response({"detail": "not found"}, status_code=404)
        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(PotpieClientError, match="404"):
                client.parse(repo_path="/tmp/repo")


# ---------------------------------------------------------------------------
# get_parsing_status()
# ---------------------------------------------------------------------------


class TestPotpieClientParsingStatus:
    def test_get_parsing_status(self):
        client = PotpieClient()
        mock_resp = _mock_response({"status": "ready", "latest": True})
        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.get_parsing_status("proj-1")
        assert result["status"] == "ready"

    def test_get_parsing_status_raises_on_404(self):
        client = PotpieClient()
        mock_resp = _mock_response({"detail": "not found"}, status_code=404)
        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(PotpieClientError):
                client.get_parsing_status("nonexistent")


# ---------------------------------------------------------------------------
# poll_parsing_status()
# ---------------------------------------------------------------------------


class TestPotpieClientPollParsingStatus:
    def test_poll_stops_when_ready(self):
        client = PotpieClient()
        responses = [
            _mock_response({"status": "submitted", "latest": False}),
            _mock_response({"status": "cloned", "latest": False}),
            _mock_response({"status": "ready", "latest": True}),
        ]
        with patch.object(client._session, "get", side_effect=responses):
            with patch("time.sleep"):
                statuses = list(client.poll_parsing_status("proj-1", poll_interval=0))

        assert len(statuses) == 3
        assert statuses[-1]["status"] == "ready"

    def test_poll_stops_on_error_status(self):
        client = PotpieClient()
        responses = [
            _mock_response({"status": "submitted", "latest": False}),
            _mock_response({"status": "error", "latest": False}),
        ]
        with patch.object(client._session, "get", side_effect=responses):
            with patch("time.sleep"):
                statuses = list(client.poll_parsing_status("proj-1", poll_interval=0))

        assert statuses[-1]["status"] == "error"

    def test_poll_stops_on_failed_status(self):
        client = PotpieClient()
        responses = [
            _mock_response({"status": "failed", "latest": False}),
        ]
        with patch.object(client._session, "get", side_effect=responses):
            with patch("time.sleep"):
                statuses = list(client.poll_parsing_status("proj-1", poll_interval=0))

        assert statuses[-1]["status"] == "failed"


# ---------------------------------------------------------------------------
# list_projects()
# ---------------------------------------------------------------------------


class TestPotpieClientListProjects:
    def test_returns_list(self):
        client = PotpieClient()
        projects = [{"id": "p1", "repo_name": "repo1"}]
        mock_resp = _mock_response(projects)
        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.list_projects()
        assert result == projects

    def test_empty_list(self):
        client = PotpieClient()
        mock_resp = _mock_response([])
        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.list_projects()
        assert result == []

    def test_raises_on_server_error(self):
        client = PotpieClient()
        mock_resp = _mock_response({"detail": "internal error"}, status_code=500)
        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(PotpieClientError):
                client.list_projects()


# ---------------------------------------------------------------------------
# list_agents()
# ---------------------------------------------------------------------------


class TestPotpieClientListAgents:
    def test_returns_list(self):
        client = PotpieClient()
        agents = [{"id": "agent1", "name": "Codebase QnA"}]
        mock_resp = _mock_response(agents)
        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.list_agents()
        assert result == agents

    def test_passes_list_system_agents_param(self):
        client = PotpieClient()
        mock_resp = _mock_response([])
        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            client.list_agents(list_system_agents=False)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["list_system_agents"] is False


# ---------------------------------------------------------------------------
# create_conversation()
# ---------------------------------------------------------------------------


class TestPotpieClientCreateConversation:
    def test_creates_conversation(self):
        client = PotpieClient()
        mock_resp = _mock_response(
            {"conversation_id": "conv-1", "message": "Conversation created"}
        )
        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            result = client.create_conversation("proj-1", "agent-1")

        assert result["conversation_id"] == "conv-1"
        _, kwargs = mock_post.call_args
        assert "proj-1" in kwargs["json"]["project_ids"]
        assert "agent-1" in kwargs["json"]["agent_ids"]

    def test_raises_on_error(self):
        client = PotpieClient()
        mock_resp = _mock_response({"detail": "project not found"}, status_code=404)
        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(PotpieClientError):
                client.create_conversation("bad-project", "agent-1")


# ---------------------------------------------------------------------------
# send_message()
# ---------------------------------------------------------------------------


class TestPotpieClientSendMessage:
    def test_send_message(self):
        client = PotpieClient()
        mock_resp = _mock_response({"message": "Hello!", "citations": []})
        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            result = client.send_message("conv-1", "What does this code do?")

        assert result["message"] == "Hello!"
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["content"] == "What does this code do?"
        assert kwargs["params"]["stream"] == "false"

    def test_send_message_stream_true(self):
        client = PotpieClient()
        mock_resp = _mock_response({"message": "streaming"})
        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            client.send_message("conv-1", "hello", stream=True)
        _, kwargs = mock_post.call_args
        assert kwargs["params"]["stream"] == "true"


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


class TestPotpieClientHealth:
    def test_health_ok(self):
        client = PotpieClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(client._session, "get", return_value=mock_resp):
            assert client.health() is True

    def test_health_server_error(self):
        client = PotpieClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch.object(client._session, "get", return_value=mock_resp):
            assert client.health() is False

    def test_health_connection_error(self):
        client = PotpieClient()
        with patch.object(
            client._session, "get", side_effect=requests.ConnectionError()
        ):
            assert client.health() is False
