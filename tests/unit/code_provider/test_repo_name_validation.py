import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.modules.code_provider.github.github_service import GithubService


pytestmark = pytest.mark.unit


class TestValidateRepoNameFormat:

    @pytest.mark.parametrize(
        "repo_name",
        [
            "owner/repo",
            "my-org/my-repo",
            "user123/project-name",
            "a/b",
            "octocat/Hello-World",
            "org/repo_with_underscores",
            "org/repo.with.dots",
            "org/repo-with-mixed_chars.v2",
            "A/B",
            "user-1/repo-2",
            "a" * 39 + "/repo",
        ],
    )
    def test_valid_repo_names(self, repo_name):
        GithubService._validate_repo_name_format(repo_name)

    @pytest.mark.parametrize(
        "repo_name",
        [
            "",
            "noslash",
            "/repo",
            "owner/",
            "owner/repo/extra",
            "owner//repo",
            "-owner/repo",
            "owner-/repo",
            "my--org/repo",
            "owner/repo name",
            "owner name/repo",
            "owner/@repo",
            "owner/repo!",
            ".owner/repo",
            "_owner/repo",
            "a" * 40 + "/repo",
            "owner/.",
            "owner/..",
            "owner/repo.git",
            "owner/my-project.git",
        ],
    )
    def test_invalid_repo_names(self, repo_name):
        with pytest.raises(ValueError, match="Invalid repository name format"):
            GithubService._validate_repo_name_format(repo_name)


class TestGetBranchListValidation:

    @pytest.fixture
    def service(self):
        with patch.object(GithubService, "__init__", lambda self, db: None):
            svc = GithubService.__new__(GithubService)
            return svc

    @pytest.mark.asyncio
    async def test_rejects_invalid_repo_name(self, service):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await service.get_branch_list("not-a-valid-name")
        assert "Invalid repository name format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_rejects_empty_owner(self, service):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await service.get_branch_list("/repo")
        assert "Invalid repository name format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_rejects_consecutive_hyphens_in_owner(self, service):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await service.get_branch_list("my--org/repo")
        assert "consecutive hyphens" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_name_proceeds_to_api(self, service):
        mock_repo = MagicMock()
        mock_repo.default_branch = "main"
        mock_branch = MagicMock()
        mock_branch.name = "dev"
        mock_repo.get_branches.return_value = [mock_branch]

        with patch.object(service, "get_repo", return_value=(MagicMock(), mock_repo)):
            result = await service.get_branch_list("owner/repo")
            assert result == {"branches": ["main", "dev"]}


class TestCheckPublicRepoValidation:

    @pytest.fixture
    def service(self):
        with patch.object(GithubService, "__init__", lambda self, db: None):
            svc = GithubService.__new__(GithubService)
            return svc

    @pytest.mark.asyncio
    async def test_rejects_invalid_repo_name(self, service):
        with pytest.raises(ValueError, match="Invalid repository name format"):
            await service.check_public_repo("not-a-valid-name")

    @pytest.mark.asyncio
    async def test_rejects_repo_ending_with_dot_git(self, service):
        with pytest.raises(ValueError, match="end with '.git'"):
            await service.check_public_repo("owner/repo.git")

    @pytest.mark.asyncio
    async def test_valid_name_proceeds_to_api(self, service):
        mock_github = MagicMock()
        with patch.object(
            GithubService, "get_public_github_instance", return_value=mock_github
        ):
            result = await service.check_public_repo("owner/repo")
            assert result is True
            mock_github.get_repo.assert_called_once_with("owner/repo")

    @pytest.mark.asyncio
    async def test_returns_false_on_api_error(self, service):
        mock_github = MagicMock()
        mock_github.get_repo.side_effect = Exception("not found")
        with patch.object(
            GithubService, "get_public_github_instance", return_value=mock_github
        ):
            result = await service.check_public_repo("owner/repo")
            assert result is False
