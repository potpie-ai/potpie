import pytest
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.main import app  # Assuming `app` is in your `main.py` file
from app.core.database import Base, get_db
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_model import Project


# Database setup for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Use a new test database for each test session
@pytest.fixture(scope="session")
def db():
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)

# Override the get_db dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# Mock authenticated user (Assuming `AuthService.check_auth` checks for user auth)
@pytest.fixture
def mock_user():
    return {"user_id": "test_user_id", "email": "test@example.com"}


# Test for parsing a new directory (POST /parse)
@pytest.mark.asyncio
async def test_parse_new_directory(db, mock_user):
    # Prepare the input for the parsing request
    repo_details = {
        "repo_name": "test_repo",
        "repo_path": "/path/to/repo",
        "branch_name": "main"
    }

    # Simulate the request
    response = await client.post("/parse", json=repo_details, headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert "project_id" in response_data
    assert response_data["status"] == ProjectStatusEnum.SUBMITTED.value

    # Validate that the project was created in the DB
    project = db.query(Project).filter_by(id=response_data["project_id"]).first()
    assert project is not None
    assert project.status == ProjectStatusEnum.SUBMITTED.value


# Test for parsing an existing directory (POST /parse)
@pytest.mark.asyncio
async def test_parse_existing_directory(db, mock_user):
    # Pre-create a project in the database to simulate an existing project
    project_id = "existing_project"
    db.add(Project(id=project_id, name="test_repo", status=ProjectStatusEnum.READY))
    db.commit()

    # Prepare the input for the parsing request
    repo_details = {
        "repo_name": "test_repo",
        "repo_path": "/path/to/repo",
        "branch_name": "main"
    }

    # Simulate the request
    response = await client.post("/parse", json=repo_details, headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["project_id"] == project_id

    # Check that the status has been updated if needed
    if not await ParseHelper(db).check_commit_status(project_id):
        assert response_data["status"] == ProjectStatusEnum.SUBMITTED.value
    else:
        assert response_data["status"] == ProjectStatusEnum.READY.value


# Test for fetching parsing status (GET /parsing-status/{project_id})
@pytest.mark.asyncio
async def test_get_parsing_status(db, mock_user):
    # Pre-create a project in the database
    project_id = "existing_project"
    db.add(Project(id=project_id, name="test_repo", status=ProjectStatusEnum.PARSED))
    db.commit()

    # Simulate the request
    response = await client.get(f"/parsing-status/{project_id}", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == ProjectStatusEnum.PARSED.value
    assert response_data["latest"] is True  # Assuming commit is up to date


# Test for a non-existent project (GET /parsing-status/{project_id})
@pytest.mark.asyncio
async def test_get_parsing_status_nonexistent_project():
    non_existent_project_id = "nonexistent_project"

    # Simulate the request
    response = await client.get(f"/parsing-status/{non_existent_project_id}", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 404
    response_data = response.json()
    assert response_data["detail"] == "Project not found"
