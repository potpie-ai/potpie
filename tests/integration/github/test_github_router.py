from unittest.mock import patch
import pytest
from app.modules.auth.auth_service import AuthService

# Mock response data
MOCK_GITHUB_ORGS_RESPONSE = []
MOCK_GITHUB_REPOS_RESPONSE = {
    "repositories": [
        {
            "id": 1,
            "name": "test-repo",
            "full_name": "test-github-user/test-repo",
            "private": False,
            "html_url": "https://github.com/test-github-user/test-repo",
            "owner": {
                "login": "test-github-user"
            }
        }
    ]
}
MOCK_INSTALLATIONS_RESPONSE = [
    {
        "id": 1,
        "account": {
            "login": "test-github-user",
            "type": "User"
        },
        "repositories_url": "https://api.github.com/installations/1/repositories"
    }
]

@pytest.fixture
def mock_auth_middleware():
    
    async def mock_check_auth():
        return {"user_id": "test-user-id"}
    
    original_check_auth = AuthService.check_auth
    AuthService.check_auth = mock_check_auth
    
    yield
    
    AuthService.check_auth = original_check_auth

def test_get_user_repos(client, db_session, mock_auth_user):
    with patch('app.modules.code_provider.github.github_service.GithubService.get_combined_user_repos') as mock_get_repos:
        # Configure mock to return test data
        mock_get_repos.return_value = MOCK_GITHUB_REPOS_RESPONSE
        
        # Make the request
        response = client.get("/api/v1/github/user-repos")
        
        print(f"Response status: {response.status_code}")  # Debug print
        print(f"Response body: {response.text}")  # Debug print
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert "repositories" in data
        assert len(data["repositories"]) > 0
        
        # Verify first repository data
        first_repo = data["repositories"][0]
        assert "name" in first_repo
        assert "full_name" in first_repo
        assert "private" in first_repo

def test_get_user_repos_unauthorized(client_without_auth):
    """Test accessing endpoint without authentication"""
    response = client_without_auth.get("/api/v1/github/user-repos")
    assert response.status_code == 401
