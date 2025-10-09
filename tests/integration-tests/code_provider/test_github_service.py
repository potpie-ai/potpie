# In tests/integration-tests/code_provider/github/test_github_service_live.py

import pytest
import os
from fastapi import HTTPException
from app.modules.code_provider.github.github_service import GithubService
from app.modules.projects.projects_model import Project

pytestmark = pytest.mark.github_live

REPO_NAME = "octocat/Hello-World"
# THE FIX: The actual filename in the repo is 'README', not 'README.md'.
README_FILENAME = "README"

@pytest.mark.asyncio
async def test_check_public_repo_live(github_service_with_fake_redis: GithubService):
    """ Verifies the service can correctly identify a public repo and a non-existent one. """
    assert await github_service_with_fake_redis.check_public_repo(REPO_NAME) is True
    assert await github_service_with_fake_redis.check_public_repo("non-existent/repo-12345") is False

@pytest.mark.asyncio
async def test_get_branch_list_live(github_service_with_fake_redis: GithubService, ensure_default_branch: str):
    """ Verifies the service can fetch the list of branches for a real public repo. """
    result = await github_service_with_fake_redis.get_branch_list(REPO_NAME)
    assert "branches" in result
    assert len(result["branches"]) > 0
    assert result["branches"][0] == ensure_default_branch

def test_get_file_content_live_full_and_slice(github_service_with_fake_redis: GithubService, ensure_default_branch: str):
    """ Verifies fetching full file content and a slice of it from the correct filename. """
    # Full content
    content_full = github_service_with_fake_redis.get_file_content(
        repo_name=REPO_NAME,
        file_path=README_FILENAME, # <-- USE THE CORRECT FILENAME
        branch_name=ensure_default_branch,
        start_line=0, end_line=0, project_id="test", commit_id=None
    )
    assert "Hello World" in content_full
    assert content_full.count("\n") >= 1 # The file has at least one newline

    # Sliced content (first 2 lines)
    content_slice = github_service_with_fake_redis.get_file_content(
        repo_name=REPO_NAME,
        file_path=README_FILENAME, # <-- USE THE CORRECT FILENAME
        branch_name=ensure_default_branch,
        start_line=1, end_line=2, project_id="test", commit_id=None
    )
    assert "Hello World" in content_slice
    assert content_slice.count("\n") <= 1
    assert content_full.startswith(content_slice)

@pytest.mark.asyncio
async def test_get_project_structure_live_cache_miss_then_hit(
    github_service_with_fake_redis: GithubService,
    hello_world_project,
    ensure_default_branch
):
    """ Verifies that caching works and that the structure contains the correct base filename. """
    # 1. First call (Cache Miss)
    structure_miss = await github_service_with_fake_redis.get_project_structure_async(
        project_id=hello_world_project.id
    )
    assert isinstance(structure_miss, str)
    # THE FIX: Assert for the base name 'README' without the extension.
    assert "README" in structure_miss
    
    # 2. Second call (Cache Hit)
    structure_hit = await github_service_with_fake_redis.get_project_structure_async(
        project_id=hello_world_project.id
    )
    assert structure_hit == structure_miss

@pytest.mark.github_live
@pytest.mark.usefixtures("require_private_repo_secrets")
class TestPrivateRepoAccess:

    def test_get_repo_can_access_private_repo(self, github_service_with_fake_redis: GithubService):
        """
        Verifies that get_repo can successfully access a private repository
        using the PAT provided in GH_TOKEN_LIST.
        """
        private_repo_name = os.environ["PRIVATE_TEST_REPO_NAME"]
        
        # ACT & ASSERT
        # This will succeed if the PAT has access. The service's fallback logic
        # (`get_public_github_instance`) is what actually uses the PAT.
        try:
            _, repo = github_service_with_fake_redis.get_repo(private_repo_name)
            assert repo.full_name.lower() == private_repo_name.lower()
            assert repo.private is True
        except HTTPException as e:
            pytest.fail(f"Failed to access private repo {private_repo_name}. "
                        f"Ensure your PAT in GH_TOKEN_LIST has access. Error: {e.detail}")

    def test_get_file_content_from_private_repo(self, github_service_with_fake_redis: GithubService):
        """
        Verifies that the service can read the content of a file from a
        configured private repository.
        """
        private_repo_name = os.environ["PRIVATE_TEST_REPO_NAME"]
        
        # ACT
        content = github_service_with_fake_redis.get_file_content(
            repo_name=private_repo_name,
            file_path="private_file.txt", # The file you created in Phase 1
            branch_name=None, # Should use repo's default branch
            start_line=0, end_line=0, project_id="test", commit_id=None
        )

        # ASSERT
        assert "This is a private file" in content

    @pytest.mark.asyncio
    async def test_get_project_structure_from_private_repo(
        self,
        github_service_with_fake_redis: GithubService,
        private_project_committed: Project
    ):
        """
        Verifies the end-to-end flow of getting the file structure for a
        private repository project stored in our database.
        """
        # ACT
        structure = await github_service_with_fake_redis.get_project_structure_async(
            project_id=private_project_committed.id
        )

        # ASSERT
        assert isinstance(structure, str)
        assert "private_file.txt" in structure