import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app  # Assuming this is where your FastAPI app is initialized
from app.modules.github.github_service import GithubService
from app.modules.users.user_service import UserService

# Use TestClient to simulate requests to the API
client = TestClient(app)

# Development database URL (use your actual development DB URL here)
DATABASE_URL = "postgresql://postgres:mysecretpassword@host.docker.internal:5432/momentum"  # Replace with your actual dev DB URL

# Set up the database session fixture using your local database
@pytest.fixture(scope="function")
def db_session():
    # Create the database engine using your development database
    engine = create_engine(DATABASE_URL)
    
    # Create a new session for testing
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    # Yield the session for use in tests
    yield session

    # Rollback any changes after each test
    session.rollback()

    # Close the session after the test
    session.close()

# Fixture to get authorized headers (fetching user from real DB)
@pytest.fixture
def authorized_headers(db_session):
    # Initialize UserService with the test DB session
    user_service = UserService(db_session)
    test_user = os.getenv("TEST_USER")
    # Fetch a user from the database (ensure you have a valid user in the DB)
    user = user_service.get_user_by_uid(test_user)  # Replace with a real user ID from your DB
    
    if not user:
        raise Exception("Test user not found in the database")
    
    # Initialize GithubService with the DB session
    github_service = GithubService(db_session)

    # Generate the JWT using GithubService.get_github_repo_details
    # This method internally handles the GitHub authentication process
    repo_name = "octocat/Hello-World"  # Replace with any repo name as we only need auth
    github, response, auth, _ = github_service.get_github_repo_details(repo_name)
    
    if response.status_code != 200:
        raise Exception("Failed to authenticate with GitHub")

    # The auth object contains the JWT token we need
    app_auth = auth.get_installation_auth(response.json()["id"])
    jwt_token = app_auth.token

    # Construct the authorized headers
    return {
        "Authorization": f"Bearer {jwt_token}"
    }
# Test Case 1: Test get_user_repos Endpoint
def test_get_user_repos(db_session, authorized_headers):
    response = client.get(
        "/api/v1/github/user-repos",
        headers=authorized_headers
    )
    
    assert response.status_code == 200
    json_data = response.json()
    assert "repositories" in json_data
    assert isinstance(json_data["repositories"], list)
    assert len(json_data["repositories"]) > 0  # Ensure some repos exist

# Test Case 2: Test get_branch_list Endpoint
def test_get_branch_list(db_session, authorized_headers):
    repo_name = "your-username/repo-name"  # Replace with a valid repo name
    
    response = client.get(
        f"/api/v1/github/get-branch-list?repo_name={repo_name}",
        headers=authorized_headers
    )
    
    assert response.status_code == 200
    json_data = response.json()
    assert "branches" in json_data
    assert isinstance(json_data["branches"], list)
    assert len(json_data["branches"]) > 0  # Ensure there are branches


# Test Case 3: Test GithubService.get_public_github_repo
def test_github_service_get_public_github_repo():
    repo_name = "octocat/Hello-World"  # Replace with a valid public repo name
    repo_details, owner = GithubService.get_public_github_repo(repo_name)
    
    assert "id" in repo_details
    assert "full_name" in repo_details
    assert repo_details["full_name"] == repo_name
