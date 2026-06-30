import pytest
from unittest.mock import patch, MagicMock
from app.modules.code_provider.gitbucket.gitbucket_provider import GitBucketProvider
from app.modules.code_provider.base.code_provider_interface import AuthMethod


class TestGitBucketProvider:
    """Test suite for GitBucket provider."""

    def test_init_requires_base_url(self):
        """Test that initialization requires base_url."""
        with pytest.raises(ValueError, match="requires base_url"):
            GitBucketProvider(base_url=None)

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from base_url."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3/")
        assert provider.base_url == "http://localhost:8080/api/v3"

    def test_supported_auth_methods(self):
        """Test supported authentication methods."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")
        methods = provider.get_supported_auth_methods()

        assert AuthMethod.PERSONAL_ACCESS_TOKEN in methods
        assert AuthMethod.BASIC_AUTH in methods
        assert AuthMethod.OAUTH_TOKEN in methods
        assert AuthMethod.APP_INSTALLATION not in methods

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_authenticate_with_pat(self, mock_github):
        """Test authentication with Personal Access Token."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        credentials = {"token": "test_token"}
        provider.authenticate(credentials, AuthMethod.PERSONAL_ACCESS_TOKEN)

        mock_github.assert_called_once_with(
            "test_token", base_url="http://localhost:8080/api/v3"
        )
        assert provider.client is not None

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_authenticate_with_basic_auth(self, mock_github):
        """Test authentication with Basic Auth."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        credentials = {"username": "user", "password": "pass"}
        provider.authenticate(credentials, AuthMethod.BASIC_AUTH)

        mock_github.assert_called_once_with(
            "user", "pass", base_url="http://localhost:8080/api/v3"
        )

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_authenticate_with_oauth(self, mock_github):
        """Test authentication with OAuth token."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        credentials = {"access_token": "oauth_token"}
        provider.authenticate(credentials, AuthMethod.OAUTH_TOKEN)

        mock_github.assert_called_once_with(
            "oauth_token", base_url="http://localhost:8080/api/v3"
        )

    def test_authenticate_app_installation_raises_error(self):
        """Test that App Installation auth raises appropriate error."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        with pytest.raises(NotImplementedError, match="does not support GitHub App"):
            provider.authenticate({}, AuthMethod.APP_INSTALLATION)

    def test_get_provider_name(self):
        """Test provider name is 'gitbucket'."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")
        assert provider.get_provider_name() == "gitbucket"

    def test_get_api_base_url(self):
        """Test getting API base URL."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")
        assert provider.get_api_base_url() == "http://localhost:8080/api/v3"

    def test_operations_require_authentication(self):
        """Test that operations require authentication."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        with pytest.raises(RuntimeError, match="not authenticated"):
            provider.get_repository("owner/repo")

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_get_repository(self, mock_github):
        """Test getting repository details."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        # Setup mock
        mock_client = MagicMock()
        mock_repo = MagicMock()
        mock_repo.id = 123
        mock_repo.name = "test-repo"
        mock_repo.full_name = "owner/test-repo"
        mock_repo.owner.login = "owner"
        mock_repo.default_branch = "master"
        mock_repo.private = False
        mock_repo.html_url = "http://localhost:8080/owner/test-repo"
        mock_repo.description = "Test repository"
        mock_repo.language = "Python"

        mock_client.get_repo.return_value = mock_repo
        provider.client = mock_client

        result = provider.get_repository("owner/test-repo")

        assert result["id"] == 123
        assert result["name"] == "test-repo"
        assert result["full_name"] == "owner/test-repo"
        assert result["owner"] == "owner"
        assert result["default_branch"] == "master"
        assert result["private"] is False

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_check_repository_access(self, mock_github):
        """Test checking repository access."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        # Setup mock
        mock_client = MagicMock()
        mock_repo = MagicMock()
        mock_repo.id = 123
        mock_repo.name = "test-repo"
        mock_repo.full_name = "owner/test-repo"
        mock_repo.owner.login = "owner"
        mock_repo.default_branch = "master"
        mock_repo.private = False
        mock_repo.html_url = "http://localhost:8080/owner/test-repo"
        mock_repo.description = "Test repository"
        mock_repo.language = "Python"

        mock_client.get_repo.return_value = mock_repo
        provider.client = mock_client

        assert provider.check_repository_access("owner/test-repo") is True

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_check_repository_access_fails(self, mock_github):
        """Test checking repository access when it fails."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.get_repo.side_effect = Exception("Access denied")
        provider.client = mock_client

        assert provider.check_repository_access("owner/test-repo") is False

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_list_branches(self, mock_github):
        """Test listing branches."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        # Setup mock
        mock_client = MagicMock()
        mock_repo = MagicMock()
        mock_repo.default_branch = "master"

        mock_branch1 = MagicMock()
        mock_branch1.name = "master"
        mock_branch2 = MagicMock()
        mock_branch2.name = "develop"
        mock_branch3 = MagicMock()
        mock_branch3.name = "feature/test"

        mock_repo.get_branches.return_value = [mock_branch2, mock_branch1, mock_branch3]
        mock_client.get_repo.return_value = mock_repo
        provider.client = mock_client

        branches = provider.list_branches("owner/test-repo")

        # Default branch should be first
        assert branches[0] == "master"
        assert "develop" in branches
        assert "feature/test" in branches
        assert len(branches) == 3

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_get_rate_limit_info(self, mock_github):
        """Test getting rate limit info."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        # Setup mock
        mock_client = MagicMock()
        mock_rate_limit = MagicMock()
        mock_rate_limit.core.limit = 5000
        mock_rate_limit.core.remaining = 4999
        mock_rate_limit.core.reset.isoformat.return_value = "2025-01-01T00:00:00"

        mock_client.get_rate_limit.return_value = mock_rate_limit
        provider.client = mock_client

        result = provider.get_rate_limit_info()

        assert result["limit"] == 5000
        assert result["remaining"] == 4999
        assert result["reset_at"] == "2025-01-01T00:00:00"

    @patch("app.modules.code_provider.gitbucket.gitbucket_provider.Github")
    def test_get_rate_limit_info_not_supported(self, mock_github):
        """Test getting rate limit info when GitBucket doesn't support it."""
        provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")

        # Setup mock to raise exception
        mock_client = MagicMock()
        from github.GithubException import GithubException

        mock_client.get_rate_limit.side_effect = GithubException(404, "Not found")
        provider.client = mock_client

        result = provider.get_rate_limit_info()

        # Should return None values when not supported
        assert result["limit"] is None
        assert result["remaining"] is None
        assert result["reset_at"] is None
