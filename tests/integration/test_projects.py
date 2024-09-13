import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app  # Assuming your FastAPI app instance is in app.main
from app.core.database import Base, get_db
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_schema import ProjectStatusEnum

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the database dependency to use the test database
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="function")
def db():
    """Set up a clean database for each test."""
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)

# Mock authenticated user (assuming AuthService.check_auth returns user information)
@pytest.fixture
def mock_user():
    return {"user_id": "test_user_id", "email": "test@example.com"}

# Helper function to add a project to the database
def create_test_project(db, project_id, repo_name, user_id, status):
    project = Project(id=project_id, repo_name=repo_name, user_id=user_id, status=status)
    db.add(project)
    db.commit()
    return project

# Test for the project list (GET /projects/list)
@pytest.mark.asyncio
async def test_get_project_list(db, mock_user):
    # Pre-create projects in the database for testing
    create_test_project(db, "project_1", "repo_1", "test_user_id", ProjectStatusEnum.SUBMITTED.value)
    create_test_project(db, "project_2", "repo_2", "test_user_id", ProjectStatusEnum.READY.value)

    # Simulate a request to fetch the project list
    response = client.get("/api/v1/projects/list", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    project_list = response.json()
    assert len(project_list) == 2
    assert project_list[0]["repo_name"] == "repo_1"
    assert project_list[1]["repo_name"] == "repo_2"

# Test for project deletion (DELETE /projects)
@pytest.mark.asyncio
async def test_delete_project(db, mock_user):
    # Pre-create a project in the database for deletion
    project = create_test_project(db, "project_1", "repo_1", "test_user_id", ProjectStatusEnum.READY.value)

    # Simulate a request to delete the project
    response = client.delete(f"/projects?project_id={project.id}", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Project deleted successfully."
    assert response_data["id"] == "project_1"

    # Check if the project is actually deleted from the database
    deleted_project = db.query(Project).filter_by(id="project_1").first()
    assert deleted_project is None

# Test for deletion of a non-existent project
@pytest.mark.asyncio
async def test_delete_non_existent_project(db, mock_user):
    # Simulate a request to delete a project that doesn't exist
    response = client.delete("/api/v1/projects?project_id=999", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 404
    response_data = response.json()
    assert response_data["detail"] == "Project not found."

# Test for empty project list (GET /projects/list)
@pytest.mark.asyncio
async def test_get_empty_project_list(db, mock_user):
    # Ensure there are no projects in the database
    response = client.get("api/v1/projects/list", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    project_list = response.json()
    assert project_list == []

# Test for project deletion with unauthorized user
@pytest.mark.asyncio
async def test_delete_project_unauthorized():
    # Simulate a request without the correct authentication header
    response = client.delete("api/v1/projects?project_id=1")

    # Validate response
    assert response.status_code == 401  # Unauthorized request
    response_data = response.json()
    assert "detail" in response_data
