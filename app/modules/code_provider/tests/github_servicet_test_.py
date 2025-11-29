import os
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException

from app.modules.code_provider.github.github_service import GithubService
from app.modules.projects.projects_service import ProjectService

@pytest.fixture
def github_service():
    mock_db = Mock()
    mock_project_service = Mock(spec=ProjectService)
    service = GithubService(mock_db)
    service.project_manager = mock_project_service
    return service

class TestGithubServiceValidation:
    def test_valid_repo_names(self, github_service):
        """Test valid repository names that should pass validation"""
        valid_names = [
            "owner/repo",
            "user-name/repo-name",
            "user_name/repo_name",
            "user123/repo123",
            "user/repo-name",
            "user-name/repo",
            "user/repo_name",
            "user_name/repo"
        ]
        
        for repo_name in valid_names:
            # Should not raise any exception
            github_service._validate_repo_name(repo_name)

    def test_invalid_repo_names(self, github_service):
        """Test invalid repository names that should raise ValueError"""
        invalid_cases = [
            ("", "Repository name must be a non-empty string"),
            ("invalid-repo", "Repository name must be in the format 'owner/repo'"),
            ("owner/repo/extra", "Repository name must be in the format 'owner/repo'"),
            ("owner/repo!", "Repository can only contain alphanumeric characters, dashes, and underscores"),
            ("-owner/repo", "Owner cannot start or end with a dash"),
            ("owner/-repo", "Repository cannot start or end with a dash"),
            ("owner/repo-", "Repository cannot start or end with a dash"),
            ("owner--name/repo", "Owner cannot contain consecutive dashes"),
            ("owner/repo--name", "Repository cannot contain consecutive dashes"),
            ("a" * 40 + "/repo", "Owner cannot be longer than 39 characters"),
            ("owner/" + "a" * 101, "Repository cannot be longer than 100 characters")
        ]
        
        for repo_name, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                github_service._validate_repo_name(repo_name)

    def test_local_repository_path(self, github_service, tmp_path):
        """Test that local repository paths are allowed without validation"""
        # Create a temporary directory
        local_repo_path = str(tmp_path)
        
        github_service._validate_repo_name(local_repo_path)

    @pytest.mark.asyncio
    async def test_check_public_repo_validation(self, github_service):
        """Test that check_public_repo properly handles validation errors"""
        with pytest.raises(HTTPException) as exc_info:
            await github_service.check_public_repo("invalid-repo")
        
        assert exc_info.value.status_code == 400
        assert "Repository name must be in the format 'owner/repo'" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_branch_list_validation(self, github_service):
        """Test that get_branch_list properly handles validation errors"""
        with pytest.raises(HTTPException) as exc_info:
            await github_service.get_branch_list("invalid-repo")
        
        assert exc_info.value.status_code == 404
        assert "Repository not found or error fetching branches" in str(exc_info.value.detail)

    def test_non_string_input(self, github_service):
        """Test that non-string inputs are properly handled"""
        invalid_inputs = [None, 123, [], {}, True]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError, match="Repository name must be a non-empty string"):
                github_service._validate_repo_name(invalid_input) 