import pytest
from fastapi.testclient import TestClient
from app.main import app  # Assuming your FastAPI app instance is in app.main


client = TestClient(app)
auth_header = {"Authorization": "Bearer Token"}

@pytest.mark.asyncio
async def test_get_project_list():
    # Simulate a request to fetch the project list
    response = client.get("/api/v1/projects/list", headers=auth_header)

    # Validate the response data
    assert response.status_code == 200  # Expecting 200 OK

@pytest.mark.asyncio
async def test_delete_project():
    # Simulate a request to delete the project
    response = client.delete(f"/api/v1/projects?project_id=0191f139-a55f-76ad-ad4b-c873b1f6fb16", headers=auth_header)

    # Validate response
    assert response.status_code == 200  # Expecting 200 OK

@pytest.mark.asyncio
async def test_delete_non_existent_project():
    # Simulate a request to delete a project that doesn't exist
    response = client.delete("/api/v1/projects?project_id=999", headers=auth_header)

    # Validate response
    assert response.status_code == 404 
