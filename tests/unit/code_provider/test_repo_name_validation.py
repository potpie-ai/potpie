import os

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


class TestCodeProviderControllerValidation:

    @pytest.fixture
    def controller(self):
        from app.modules.code_provider.code_provider_controller import CodeProviderController
        with patch.object(CodeProviderController, "__init__", lambda self, db: None):
            ctrl = CodeProviderController.__new__(CodeProviderController)
            ctrl.branch_cache = MagicMock()
            ctrl.branch_cache.get_branches_async = AsyncMock(return_value=None)
            return ctrl

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CODE_PROVIDER": "github"})
    async def test_get_branch_list_rejects_invalid_name(self, controller):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await controller.get_branch_list("invalid-no-slash")
        assert exc_info.value.status_code == 400
        assert "Invalid repository name format" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CODE_PROVIDER": "github"})
    async def test_get_branch_list_rejects_consecutive_hyphens(self, controller):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await controller.get_branch_list("my--org/repo")
        assert exc_info.value.status_code == 400
        assert "consecutive hyphens" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CODE_PROVIDER": "github"})
    async def test_get_branch_list_rejects_dot_git(self, controller):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await controller.get_branch_list("owner/repo.git")
        assert exc_info.value.status_code == 400
        assert ".git" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CODE_PROVIDER": "github"})
    async def test_check_public_repo_rejects_invalid_name(self, controller):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await controller.check_public_repo("nope")
        assert exc_info.value.status_code == 400
        assert "Invalid repository name format" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CODE_PROVIDER": "github"})
    async def test_check_public_repo_rejects_reserved_name(self, controller):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await controller.check_public_repo("owner/.")
        assert exc_info.value.status_code == 400
        assert "must not be" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CODE_PROVIDER": "gitbucket"})
    async def test_get_branch_list_skips_validation_for_non_github(self, controller):
        controller.branch_cache.get_branches_async = AsyncMock(return_value=["main"])
        controller._filter_branches = MagicMock(return_value=["main"])
        controller._paginate_branches = MagicMock(return_value={"branches": ["main"]})

        result = await controller.get_branch_list("anything-goes-here")
        assert result == {"branches": ["main"]}
