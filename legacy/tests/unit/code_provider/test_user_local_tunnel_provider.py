"""Unit tests for UserLocalTunnelProvider."""

from unittest.mock import MagicMock, patch

import pytest

from app.modules.code_provider.base.code_provider_interface import AuthMethod
from app.modules.code_provider.user_local_tunnel.user_local_tunnel_provider import (
    UserLocalTunnelProvider,
)


pytestmark = pytest.mark.unit


class TestInit:
    """Tests for __init__ method."""

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        provider = UserLocalTunnelProvider(
            user_id="user123",
            conversation_id="conv456",
            tunnel_url="https://tunnel.example.com",
        )
        assert provider.user_id == "user123"
        assert provider.conversation_id == "conv456"
        assert provider.tunnel_url == "https://tunnel.example.com"
        assert provider.client is None

    def test_init_without_tunnel_url(self):
        """Test initialization without tunnel_url."""
        provider = UserLocalTunnelProvider(
            user_id="user123",
            conversation_id="conv456",
        )
        assert provider.user_id == "user123"
        assert provider.conversation_id == "conv456"
        assert provider.tunnel_url is None
        assert provider.client is None

    def test_init_without_user_id(self):
        """Test initialization without user_id."""
        provider = UserLocalTunnelProvider(
            tunnel_url="https://tunnel.example.com",
        )
        assert provider.user_id is None
        assert provider.tunnel_url == "https://tunnel.example.com"
        assert provider.client is None

    def test_init_with_no_params(self):
        """Test initialization with no parameters."""
        provider = UserLocalTunnelProvider()
        assert provider.user_id is None
        assert provider.conversation_id is None
        assert provider.tunnel_url is None
        assert provider.client is None


class TestGetTunnelUrl:
    """Tests for _get_tunnel_url method."""

    def test_provided_url_returned(self):
        """Test that provided tunnel_url is returned directly."""
        provider = UserLocalTunnelProvider(
            user_id="user123",
            tunnel_url="https://tunnel.example.com",
        )
        result = provider._get_tunnel_url()
        assert result == "https://tunnel.example.com"

    def test_user_id_missing_returns_none(self):
        """Test that None is returned when user_id is missing and no tunnel_url."""
        provider = UserLocalTunnelProvider()
        result = provider._get_tunnel_url()
        assert result is None

    @patch("app.modules.tunnel.tunnel_service.get_tunnel_service")
    def test_tunnel_service_success(self, mock_get_tunnel_service):
        """Test successful tunnel URL lookup via tunnel service."""
        mock_tunnel_service = MagicMock()
        mock_tunnel_service.get_tunnel_url.return_value = "https://tunnel.service.com"
        mock_get_tunnel_service.return_value = mock_tunnel_service

        provider = UserLocalTunnelProvider(user_id="user123")
        result = provider._get_tunnel_url()

        assert result == "https://tunnel.service.com"
        assert provider.tunnel_url == "https://tunnel.service.com"
        mock_tunnel_service.get_tunnel_url.assert_called_once_with("user123", None)

    @patch("app.modules.tunnel.tunnel_service.get_tunnel_service")
    def test_tunnel_service_returns_none(self, mock_get_tunnel_service):
        """Test when tunnel service returns None."""
        mock_tunnel_service = MagicMock()
        mock_tunnel_service.get_tunnel_url.return_value = None
        mock_get_tunnel_service.return_value = mock_tunnel_service

        provider = UserLocalTunnelProvider(user_id="user123")
        result = provider._get_tunnel_url()

        assert result is None
        assert provider.tunnel_url is None

    @patch("app.modules.tunnel.tunnel_service.get_tunnel_service")
    def test_tunnel_service_exception_logs_warning(self, mock_get_tunnel_service):
        """Test that exceptions from tunnel service are caught and logged."""
        mock_get_tunnel_service.side_effect = Exception("Service unavailable")

        provider = UserLocalTunnelProvider(user_id="user123")
        result = provider._get_tunnel_url()

        assert result is None


class TestMakeTunnelRequest:
    """Tests for _make_tunnel_request method."""

    def test_tunnel_unavailable(self):
        """Test when tunnel URL is not available."""
        provider = UserLocalTunnelProvider()
        result = provider._make_tunnel_request("GET", "/api/test")
        assert result is None

    @patch("httpx.Client")
    def test_successful_get_200(self, mock_client_class):
        """Test successful GET request returning 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "test"}

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider._make_tunnel_request("GET", "/api/test")

        assert result == {"success": True, "data": "test"}
        mock_client_instance.get.assert_called_once()

    @patch("httpx.Client")
    def test_successful_post_200(self, mock_client_class):
        """Test successful POST request returning 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "id": 123}

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider._make_tunnel_request(
            "POST", "/api/create", json_data={"name": "test"}
        )

        assert result == {"success": True, "id": 123}
        mock_client_instance.post.assert_called_once()

    @patch("httpx.Client")
    def test_non_200_status_404(self, mock_client_class):
        """Test 404 status code returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider._make_tunnel_request("GET", "/api/test")

        assert result is None

    @patch("httpx.Client")
    def test_non_200_status_500(self, mock_client_class):
        """Test 500 status code returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider._make_tunnel_request("GET", "/api/test")

        assert result is None

    @patch("httpx.Client")
    def test_httpx_timeout_exception(self, mock_client_class):
        """Test httpx TimeoutException is caught."""
        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = Exception("timeout")
        mock_client_class.return_value.__enter__.return_value = mock_client_instance

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider._make_tunnel_request("GET", "/api/test")

        assert result is None

    @patch("httpx.Client")
    def test_httpx_connect_error(self, mock_client_class):
        """Test httpx ConnectError is caught."""
        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = Exception("Connection refused")
        mock_client_class.return_value.__enter__.return_value = mock_client_instance

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider._make_tunnel_request("GET", "/api/test")

        assert result is None

    def test_tunnel_url_none(self):
        """Test when tunnel URL is None (no user_id, no tunnel_url)."""
        provider = UserLocalTunnelProvider()
        result = provider._make_tunnel_request("GET", "/api/test")
        assert result is None

    @patch("httpx.Client")
    def test_request_with_params(self, mock_client_class):
        """Test request with query parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client_instance

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider._make_tunnel_request(
            "GET", "/api/test", params={"path": "file.py", "ref": "main"}
        )

        assert result == {"success": True}
        call_args = mock_client_instance.get.call_args
        assert "https://tunnel.example.com/api/test?path=file.py&ref=main" in str(
            call_args
        )


class TestGetFileContent:
    """Tests for get_file_content method."""

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_full_content(self, mock_tunnel_request):
        """Test fetching full file content without line slicing."""
        mock_tunnel_request.return_value = {
            "success": True,
            "content": "line1\nline2\nline3",
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_file_content("repo", "test.py")

        assert result == "line1\nline2\nline3"
        mock_tunnel_request.assert_called_once_with(
            "GET", "/api/files/read", params={"path": "test.py"}
        )

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_line_slicing_with_start_line(self, mock_tunnel_request):
        """Test file content with start_line only."""
        mock_tunnel_request.return_value = {
            "success": True,
            "content": "line1\nline2\nline3\nline4\nline5",
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_file_content("repo", "test.py", start_line=2)

        assert result == "line2\nline3\nline4\nline5"

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_line_slicing_with_end_line(self, mock_tunnel_request):
        """Test file content with end_line only."""
        mock_tunnel_request.return_value = {
            "success": True,
            "content": "line1\nline2\nline3\nline4\nline5",
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_file_content("repo", "test.py", end_line=3)

        assert result == "line1\nline2\nline3"

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_line_slicing_with_start_and_end_line(self, mock_tunnel_request):
        """Test file content with both start_line and end_line."""
        mock_tunnel_request.return_value = {
            "success": True,
            "content": "line1\nline2\nline3\nline4\nline5",
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_file_content("repo", "test.py", start_line=2, end_line=4)

        assert result == "line2\nline3\nline4"

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_success_false_raises_file_not_found_error(self, mock_tunnel_request):
        """Test that success:False raises FileNotFoundError."""
        mock_tunnel_request.return_value = {
            "success": False,
            "error": "File not found",
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        with pytest.raises(
            FileNotFoundError, match="Failed to read file 'test.py': File not found"
        ):
            provider.get_file_content("repo", "test.py")

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_empty_response_raises_file_not_found_error(self, mock_tunnel_request):
        """Test that empty response raises FileNotFoundError."""
        mock_tunnel_request.return_value = None

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        with pytest.raises(
            FileNotFoundError,
            match="Failed to read file 'test.py': No response from tunnel",
        ):
            provider.get_file_content("repo", "test.py")

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_malformed_response_no_success_key(self, mock_tunnel_request):
        """Test that response without success key raises FileNotFoundError."""
        mock_tunnel_request.return_value = {"content": "some content"}

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        with pytest.raises(
            FileNotFoundError, match="Failed to read file 'test.py': Unknown error"
        ):
            provider.get_file_content("repo", "test.py")

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_empty_content(self, mock_tunnel_request):
        """Test file with empty content."""
        mock_tunnel_request.return_value = {
            "success": True,
            "content": "",
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_file_content("repo", "empty.py")

        assert result == ""


class TestGetRepositoryStructure:
    """Tests for get_repository_structure method."""

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_formatting_nested_dict_to_indented_string(self, mock_tunnel_request):
        """Test formatting nested dict structure to indented string."""
        mock_tunnel_request.return_value = {
            "success": True,
            "structure": {
                "name": "root",
                "type": "directory",
                "children": [
                    {"name": "file1.py", "type": "file"},
                    {
                        "name": "subdir",
                        "type": "directory",
                        "children": [{"name": "file2.py", "type": "file"}],
                    },
                ],
            },
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_repository_structure("repo")

        assert len(result) == 1
        assert result[0]["path"] == ""
        structure_str = result[0]["structure"]
        assert "file1.py" in structure_str
        assert "subdir" in structure_str
        assert "file2.py" in structure_str

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_sorting_dirs_before_files(self, mock_tunnel_request):
        """Test that directories are sorted before files alphabetically."""
        mock_tunnel_request.return_value = {
            "success": True,
            "structure": {
                "name": "root",
                "type": "directory",
                "children": [
                    {"name": "zfile.txt", "type": "file"},
                    {"name": "adir", "type": "directory", "children": []},
                    {"name": "bfile.txt", "type": "file"},
                ],
            },
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_repository_structure("repo")

        structure_str = result[0]["structure"]
        # adir should come before bfile.txt which should come before zfile.txt
        lines = structure_str.split("\n")
        names = [line.strip() for line in lines if line.strip()]
        adir_idx = names.index("adir")
        bfile_idx = names.index("bfile.txt")
        zfile_idx = names.index("zfile.txt")
        assert adir_idx < bfile_idx < zfile_idx

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_empty_structure(self, mock_tunnel_request):
        """Test with empty structure."""
        mock_tunnel_request.return_value = {
            "success": True,
            "structure": {},
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_repository_structure("repo")

        assert result == []

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_root_node_processing(self, mock_tunnel_request):
        """Test that root node name is not included in output."""
        mock_tunnel_request.return_value = {
            "success": True,
            "structure": {
                "name": "workspace",
                "type": "directory",
                "children": [
                    {"name": "readme.md", "type": "file"},
                ],
            },
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_repository_structure("repo")

        structure_str = result[0]["structure"]
        # Root name "workspace" should not appear, but "readme.md" should
        assert "workspace" not in structure_str
        assert "readme.md" in structure_str

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_success_false_returns_empty_list(self, mock_tunnel_request):
        """Test that success:False returns empty list."""
        mock_tunnel_request.return_value = {
            "success": False,
            "error": "Access denied",
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_repository_structure("repo")

        assert result == []

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_with_path_param(self, mock_tunnel_request):
        """Test get_repository_structure with path parameter."""
        mock_tunnel_request.return_value = {
            "success": True,
            "structure": {
                "name": "subdir",
                "type": "directory",
                "children": [],
            },
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        result = provider.get_repository_structure("repo", path="subdir")

        assert result[0]["path"] == "subdir"
        mock_tunnel_request.assert_called_once_with(
            "GET", "/api/files/structure", params={"path": "subdir", "max_depth": "4"}
        )

    @patch.object(UserLocalTunnelProvider, "_make_tunnel_request")
    def test_with_max_depth(self, mock_tunnel_request):
        """Test get_repository_structure with max_depth parameter."""
        mock_tunnel_request.return_value = {
            "success": True,
            "structure": {
                "name": "root",
                "type": "directory",
                "children": [],
            },
        }

        provider = UserLocalTunnelProvider(tunnel_url="https://tunnel.example.com")
        provider.get_repository_structure("repo", max_depth=2)

        mock_tunnel_request.assert_called_once_with(
            "GET", "/api/files/structure", params={"max_depth": "2"}
        )


class TestNotImplementedErrorMethods:
    """Tests for methods that should raise NotImplementedError."""

    def test_create_branch_raises_not_implemented(self):
        """Test create_branch raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="Branch operations not supported"
        ):
            provider.create_branch("repo", "new-branch", "main")

    def test_compare_branches_raises_not_implemented(self):
        """Test compare_branches raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="Branch operations not supported"
        ):
            provider.compare_branches("repo", "main", "feature")

    def test_list_pull_requests_raises_not_implemented(self):
        """Test list_pull_requests raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="Pull request operations not supported"
        ):
            provider.list_pull_requests("repo")

    def test_get_pull_request_raises_not_implemented(self):
        """Test get_pull_request raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="Pull request operations not supported"
        ):
            provider.get_pull_request("repo", 123)

    def test_create_pull_request_raises_not_implemented(self):
        """Test create_pull_request raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="Pull request operations not supported"
        ):
            provider.create_pull_request("repo", "title", "body", "head", "base")

    def test_add_pull_request_comment_raises_not_implemented(self):
        """Test add_pull_request_comment raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="Pull request operations not supported"
        ):
            provider.add_pull_request_comment("repo", 123, "comment")

    def test_create_pull_request_review_raises_not_implemented(self):
        """Test create_pull_request_review raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="Pull request operations not supported"
        ):
            provider.create_pull_request_review("repo", 123, "body", "APPROVE")

    def test_list_issues_raises_not_implemented(self):
        """Test list_issues raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(NotImplementedError, match="Issue operations not supported"):
            provider.list_issues("repo")

    def test_get_issue_raises_not_implemented(self):
        """Test get_issue raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(NotImplementedError, match="Issue operations not supported"):
            provider.get_issue("repo", 123)

    def test_create_issue_raises_not_implemented(self):
        """Test create_issue raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(NotImplementedError, match="Issue operations not supported"):
            provider.create_issue("repo", "title", "body")

    def test_create_or_update_file_raises_not_implemented(self):
        """Test create_or_update_file raises NotImplementedError."""
        provider = UserLocalTunnelProvider()
        with pytest.raises(
            NotImplementedError, match="File modification operations not supported"
        ):
            provider.create_or_update_file(
                "repo", "path", "content", "message", "branch"
            )


class TestOtherMethods:
    """Tests for other provider methods."""

    def test_authenticate_returns_none(self):
        """Test authenticate returns None (no auth needed)."""
        provider = UserLocalTunnelProvider()
        result = provider.authenticate({"token": "test"}, AuthMethod.OAUTH_TOKEN)
        assert result is None

    def test_get_supported_auth_methods_returns_empty_list(self):
        """Test get_supported_auth_methods returns empty list."""
        provider = UserLocalTunnelProvider()
        result = provider.get_supported_auth_methods()
        assert result == []

    def test_get_repository(self):
        """Test get_repository returns local workspace metadata."""
        provider = UserLocalTunnelProvider()
        result = provider.get_repository("any-repo")

        assert result["id"] == "local-workspace"
        assert result["name"] == "local-workspace"
        assert result["private"] is True
        assert result["default_branch"] == "main"

    @patch.object(UserLocalTunnelProvider, "_get_tunnel_url")
    def test_check_repository_access_true(self, mock_get_tunnel_url):
        """Test check_repository_access returns True when tunnel works."""
        mock_get_tunnel_url.return_value = "https://tunnel.example.com"

        with patch.object(
            UserLocalTunnelProvider, "_make_tunnel_request"
        ) as mock_request:
            mock_request.return_value = {"success": True}
            provider = UserLocalTunnelProvider()
            result = provider.check_repository_access("repo")
            assert result is True

    @patch.object(UserLocalTunnelProvider, "_get_tunnel_url")
    def test_check_repository_access_false_no_tunnel(self, mock_get_tunnel_url):
        """Test check_repository_access returns False when no tunnel."""
        mock_get_tunnel_url.return_value = None
        provider = UserLocalTunnelProvider()
        result = provider.check_repository_access("repo")
        assert result is False

    @patch.object(UserLocalTunnelProvider, "_get_tunnel_url")
    def test_check_repository_access_false_bad_response(self, mock_get_tunnel_url):
        """Test check_repository_access returns False when response is bad."""
        mock_get_tunnel_url.return_value = "https://tunnel.example.com"

        with patch.object(
            UserLocalTunnelProvider, "_make_tunnel_request"
        ) as mock_request:
            mock_request.return_value = {"success": False}
            provider = UserLocalTunnelProvider()
            result = provider.check_repository_access("repo")
            assert result is False

    def test_list_branches_returns_main(self):
        """Test list_branches returns main branch."""
        provider = UserLocalTunnelProvider()
        result = provider.list_branches("repo")
        assert result == ["main"]

    def test_get_branch_returns_local_info(self):
        """Test get_branch returns local branch info."""
        provider = UserLocalTunnelProvider()
        result = provider.get_branch("repo", "feature-branch")

        assert result["name"] == "feature-branch"
        assert result["commit_sha"] == "local"
        assert result["protected"] is False

    def test_list_user_repositories_returns_empty(self):
        """Test list_user_repositories returns empty list."""
        provider = UserLocalTunnelProvider()
        result = provider.list_user_repositories()
        assert result == []

    def test_get_user_organizations_returns_empty(self):
        """Test get_user_organizations returns empty list."""
        provider = UserLocalTunnelProvider()
        result = provider.get_user_organizations()
        assert result == []

    def test_get_provider_name(self):
        """Test get_provider_name returns correct name."""
        provider = UserLocalTunnelProvider()
        assert provider.get_provider_name() == "user_local_tunnel"

    @patch.object(UserLocalTunnelProvider, "_get_tunnel_url")
    def test_get_api_base_url_with_tunnel(self, mock_get_tunnel_url):
        """Test get_api_base_url returns tunnel URL."""
        mock_get_tunnel_url.return_value = "https://tunnel.example.com"
        provider = UserLocalTunnelProvider()
        assert provider.get_api_base_url() == "https://tunnel.example.com"

    @patch.object(UserLocalTunnelProvider, "_get_tunnel_url")
    def test_get_api_base_url_without_tunnel(self, mock_get_tunnel_url):
        """Test get_api_base_url returns default when no tunnel."""
        mock_get_tunnel_url.return_value = None
        provider = UserLocalTunnelProvider()
        assert provider.get_api_base_url() == "tunnel://local"

    def test_get_rate_limit_info(self):
        """Test get_rate_limit_info returns infinite limits."""
        provider = UserLocalTunnelProvider()
        result = provider.get_rate_limit_info()

        assert result["limit"] == float("inf")
        assert result["remaining"] == float("inf")
        assert result["reset_at"] is None
