from unittest.mock import patch

import pytest
from fastapi import HTTPException
from app.modules.code_provider.github.github_service import GithubService

@pytest.fixture
def github_service():
    # Assuming a mock database session is provided
    db_session = None
    return GithubService(db_session)

def test_get_file_content_authenticated_success(github_service):
    # Setup mock for authenticated access
    # Call get_file_content with valid parameters
    content = github_service.get_file_content(
        repo_name="dhirenmathur/coming_soon",
        file_path="README.md",
        start_line=1,
        end_line=1,
        branch_name="main",
        project_id="123"
    )
    assert content == "# coming_soon"

def test_get_file_content_fallback_to_public(github_service):
    # Setup mock for authenticated access failure and public access success
    # Call get_file_content with valid parameters
    content = github_service.get_file_content(
        repo_name="potpie-ai/potpie",
        file_path="start.sh",
        start_line=16,
        end_line=17,
        branch_name="main",
        project_id="123"
    )
    assert content == '\necho "Starting Docker Compose..."\ndocker compose up -d'

def test_get_file_content_directory_error(github_service):
    # Setup mock for directory path
    with pytest.raises(HTTPException) as exc_info:
        github_service.get_file_content(
            repo_name="potpie-ai/potpie",
            file_path="app",
            start_line=1,
            end_line=5,
            branch_name="main",
            project_id="123"
        )
    assert exc_info.value.status_code == 400

def test_get_file_content_not_found_error(github_service):
    # Setup mock for file not found
    with pytest.raises(HTTPException) as exc_info:
        github_service.get_file_content(
            repo_name="potpie-ai/potpie",
            file_path="invalid/path",
            start_line=1,
            end_line=5,
            branch_name="main",
            project_id="123"
        )
    assert exc_info.value.status_code == 404


def test_get_file_content_invalid_encoding(github_service):
    # Mock the _detect_encoding method to simulate invalid encoding detection
    with patch.object(GithubService, '_detect_encoding', side_effect=HTTPException(status_code=400, detail="Unable to determine file encoding or low confidence")):
        with pytest.raises(HTTPException) as exc_info:
            github_service.get_file_content(
                repo_name="potpie-ai/potpie",
                file_path="start.sh",
                start_line=16,
                end_line=17,
                branch_name="main",
                project_id="123"
            )
        assert exc_info.value.status_code == 500
        assert "Unable to determine file encoding or low confidence" in exc_info.value.detail


def test_get_file_content_full_content(github_service):
    # Setup mock for full content
    content = github_service.get_file_content(
        repo_name="dhirenmathur/coming_soon",
        file_path="README.md",
        start_line=0,
        end_line=0,
        branch_name="main",
        project_id="123"
    )
    assert content == "# coming_soon"