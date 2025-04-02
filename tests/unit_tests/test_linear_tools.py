import pytest
import os
from unittest.mock import patch, MagicMock
from app.modules.intelligence.tools.linear_tools.linear_client import (
    LinearClient,
    LinearClientConfig,
)
from app.modules.intelligence.tools.linear_tools.get_linear_issue_tool import (
    get_linear_issue,
)
from app.modules.intelligence.tools.linear_tools.update_linear_issue_tool import (
    update_linear_issue,
)

# Mock responses
MOCK_ISSUE_RESPONSE = {
    "id": "TEST-123",
    "title": "Test Issue",
    "description": "Test Description",
    "state": {"id": "state1", "name": "In Progress"},
    "assignee": {"id": "user1", "name": "John Doe"},
    "team": {"id": "team1", "name": "Engineering"},
    "priority": 2,
    "url": "https://linear.app/test/issue/TEST-123",
    "createdAt": "2024-03-31T00:00:00Z",
    "updatedAt": "2024-03-31T01:00:00Z",
}

MOCK_UPDATE_RESPONSE = {
    "success": True,
    "issue": {
        "id": "TEST-123",
        "title": "Updated Title",
        "description": "Updated Description",
        "state": {"id": "state2", "name": "Done"},
        "assignee": {"id": "user2", "name": "Jane Smith"},
        "priority": 1,
        "updatedAt": "2024-03-31T02:00:00Z",
    },
}

MOCK_COMMENT_RESPONSE = {
    "success": True,
    "comment": {
        "id": "comment1",
        "body": "Test comment",
        "createdAt": "2024-03-31T03:00:00Z",
        "user": {"id": "user1", "name": "John Doe"},
    },
}


@pytest.fixture
def mock_linear_client():
    with patch(
        "app.modules.intelligence.tools.linear_tools.linear_client.LinearClient"
    ) as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_env_api_key():
    with patch.dict(os.environ, {"LINEAR_API_KEY": "test_api_key"}):
        yield


class TestLinearClient:
    def test_init(self):
        client = LinearClient("test_api_key")
        assert client.api_key == "test_api_key"
        assert client.headers == {
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
        }

    @patch("requests.post")
    def test_execute_query_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"data": {"test": "success"}}

        client = LinearClient("test_api_key")
        result = client.execute_query("query { test }")

        assert result == {"test": "success"}
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_execute_query_error(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"

        client = LinearClient("test_api_key")
        with pytest.raises(Exception) as exc_info:
            client.execute_query("query { test }")

        assert "Request failed with status code 400" in str(exc_info.value)


class TestLinearClientConfig:
    def test_singleton_pattern(self, mock_env_api_key):
        config1 = LinearClientConfig()
        config2 = LinearClientConfig()
        assert config1 is config2

    def test_missing_api_key(self):
        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LinearClientConfig()
            assert "LINEAR_API_KEY environment variable is not set" in str(
                exc_info.value
            )


class TestGetLinearIssueTool:
    def test_get_issue_success(self, mock_linear_client):
        mock_linear_client.get_issue.return_value = MOCK_ISSUE_RESPONSE

        result = get_linear_issue("TEST-123")

        assert result["id"] == "TEST-123"
        assert result["title"] == "Test Issue"
        assert result["status"] == "In Progress"
        assert result["assignee"] == "John Doe"
        assert result["team"] == "Engineering"
        assert result["priority"] == 2

    def test_get_issue_error(self, mock_linear_client):
        mock_linear_client.get_issue.side_effect = Exception("API Error")

        with pytest.raises(ValueError) as exc_info:
            get_linear_issue("TEST-123")

        assert "Error fetching Linear issue" in str(exc_info.value)


class TestUpdateLinearIssueTool:
    def test_update_issue_success(self, mock_linear_client):
        mock_linear_client.update_issue.return_value = MOCK_UPDATE_RESPONSE
        mock_linear_client.get_issue.return_value = MOCK_UPDATE_RESPONSE["issue"]

        result = update_linear_issue(
            issue_id="TEST-123",
            title="Updated Title",
            description="Updated Description",
            status="state2",
            priority=1,
        )

        assert result["id"] == "TEST-123"
        assert result["title"] == "Updated Title"
        assert result["description"] == "Updated Description"
        assert result["priority"] == 1
        assert result["comment_added"] is False

    def test_update_issue_with_comment(self, mock_linear_client):
        mock_linear_client.update_issue.return_value = MOCK_UPDATE_RESPONSE
        mock_linear_client.get_issue.return_value = MOCK_UPDATE_RESPONSE["issue"]
        mock_linear_client.comment_create.return_value = MOCK_COMMENT_RESPONSE

        result = update_linear_issue(
            issue_id="TEST-123", title="Updated Title", comment="Test comment"
        )

        assert result["id"] == "TEST-123"
        assert result["title"] == "Updated Title"
        assert result["comment_added"] is True

    def test_update_issue_error(self, mock_linear_client):
        mock_linear_client.update_issue.side_effect = Exception("API Error")

        with pytest.raises(ValueError) as exc_info:
            update_linear_issue("TEST-123", title="Updated Title")

        assert "Error updating Linear issue" in str(exc_info.value)

    def test_update_issue_no_changes(self, mock_linear_client):
        mock_linear_client.get_issue.return_value = MOCK_ISSUE_RESPONSE

        result = update_linear_issue("TEST-123")

        assert result["id"] == "TEST-123"
        assert result["title"] == "Test Issue"
        assert result["description"] == "Test Description"
        assert result["comment_added"] is False
        mock_linear_client.update_issue.assert_not_called()
