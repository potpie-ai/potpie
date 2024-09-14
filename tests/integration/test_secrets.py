import os
import pytest
from fastapi.testclient import TestClient
from google.cloud import secretmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.main import app
from app.modules.key_management.secrets_schema import CreateSecretRequest, UpdateSecretRequest

# Use the PostgreSQL connection string for local database
SQLALCHEMY_DATABASE_URL = os.getenv("POSTGRES_SERVER")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
auth_header = {
    "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjAyMTAwNzE2ZmRkOTA0ZTViNGQ0OTExNmZmNWRiZGZjOTg5OTk0MDEiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiUmFqIFV0c28iLCJwaWN0dXJlIjoiaHR0cHM6Ly9hdmF0YXJzLmdpdGh1YnVzZXJjb250ZW50LmNvbS91LzQ5ODczMjk1P3Y9NCIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS9tb21lbnR1bWNvcmUtNDZlZWEiLCJhdWQiOiJtb21lbnR1bWNvcmUtNDZlZWEiLCJhdXRoX3RpbWUiOjE3MjU2ODk2NzMsInVzZXJfaWQiOiJFelVLWWUzdHBNWFVJejdKYkdFVllrNG5mRTEyIiwic3ViIjoiRXpVS1llM3RwTVhVSXo3SmJHRVZZazRuZkUxMiIsImlhdCI6MTcyNjMyMDQwNywiZXhwIjoxNzI2MzI0MDA3LCJlbWFpbCI6ImJyYWp1dHNvQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJnaXRodWIuY29tIjpbIjQ5ODczMjk1Il0sImVtYWlsIjpbImJyYWp1dHNvQGdtYWlsLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6ImdpdGh1Yi5jb20ifX0.r4RS4PCDP9L7gcBiBPbSiSTAl-wjUWcdpTgQBkC-KLur0B-g9ZXr9e0AN5Mt_nExpPs27Jx2DlIopu3_2UTGTNuQK_u4LiqOQwSRPCleTj5uxjV_OmkgiNxNsTIPWCzmFlcqwJ4kxmyRphfWHbFinZnpon02f_v1iNALNOAw-HyXquI7hQvSxOUhZfAMVvkJfk8qnErBtIAHt2vyVRN8_qov90iBZLcVQnnnF9peSMzFRmKeRxWaUOeAy_NDy9fmpi7TJFzYBWZyBnuhDY92Ajn5IPyh4PiZHM8U9Xc0FNEF8NYh-T1E044XmjYPDTNYiBV-SMNsEJ43-EMoDAWtjw"
}

# Override the get_db dependency to use the test database
@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    try:
        yield TestingSessionLocal()
    finally:
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client():
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client

# Mocking GCP Secret Manager
class MockSecretManagerClient:
    def __init__(self):
        self.secrets = {}

    def create_secret(self, request):
        secret_id = request["secret_id"]
        self.secrets[secret_id] = request["secret"]
        return {"name": f"projects/test_project/secrets/{secret_id}"}

    def add_secret_version(self, request):
        parent = request["parent"]
        secret_id = parent.split("/")[-1]
        if secret_id not in self.secrets:
            raise Exception("Secret not found")
        self.secrets[secret_id]["payload"] = request["payload"]

    def access_secret_version(self, request):
        secret_id = request["name"].split("/")[-2]
        if secret_id not in self.secrets:
            raise Exception("Secret not found")
        return {"payload": self.secrets[secret_id]["payload"]}

    def delete_secret(self, request):
        secret_id = request["name"].split("/")[-1]
        if secret_id not in self.secrets:
            raise Exception("Secret not found")
        del self.secrets[secret_id]

@pytest.fixture
def mock_secret_manager(monkeypatch):
    mock_client = MockSecretManagerClient()

    def mock_get_client_and_project():
        return mock_client, "test_project"

    monkeypatch.setattr(
        "app.modules.key_management.secret_manager.SecretManager.get_client_and_project",
        mock_get_client_and_project,
    )

# Test creating a secret
def test_create_secret(client, mock_secret_manager):
    secret_request = {
        "provider": "openai",
        "api_key": "sk-test-1234567890abcdef1234567890abcdef1234567890abcdef",
    }

    response = client.post("/api/v1/secrets", json=secret_request, headers=auth_header)
    assert response.status_code == 200
    assert response.json() == {"message": "Secret created successfully"}

# Test fetching a secret
def test_get_secret(client, mock_secret_manager):
    response = client.get("/api/v1/secrets/openai", headers=auth_header)
    assert response.status_code == 200
    assert response.json() == {
        "api_key": "sk-test-1234567890abcdef1234567890abcdef1234567890abcdef"
    }

# Test updating a secret
def test_update_secret(client, mock_secret_manager):
    update_request = {
        "provider": "openai",
        "api_key": "sk-test-updated-1234567890abcdef1234567890abcdef",
    }

    response = client.put("/api/v1/secrets/", json=update_request, headers=auth_header)
    assert response.status_code == 200
    assert response.json() == {"message": "Secret updated successfully"}

    # Verify the updated secret
    response = client.get("/api/v1/secrets/openai", headers=auth_header)
    assert response.status_code == 200
    assert response.json() == {
        "api_key": "sk-test-updated-1234567890abcdef1234567890abcdef"
    }

# Test deleting a secret
def test_delete_secret(client, mock_secret_manager):
    response = client.delete("/api/v1/secrets/openai", headers=auth_header)
    assert response.status_code == 200
    assert response.json() == {"message": "Secret deleted successfully"}

    # Verify the secret has been deleted
    response = client.get("/api/v1/secrets/openai", headers=auth_header)
    assert response.status_code == 404
    assert response.json() == {"detail": "Secret not found in GCP Secret Manager: Secret not found"}
