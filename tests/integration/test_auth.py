import pytest
from fastapi.testclient import TestClient
from app.main import app  # Adjust this path to match your app

client = TestClient(app)

# Sample data for login and signup
login_request_data = {
    "email": "testuser@example.com",
    "password": "password123"
}

signup_request_data = {
    "uid": "test-uid-123",
    "email": "testuser@example.com",
    "displayName": "Test User",
    "emailVerified": False,
    "isAnonymous": False,
    "providerData": [
    {
      "providerId": "github.com",
      "uid": "19893222",
      "displayName": "Raj Utso",
      "email": "namikazerajutso01@gmail.com",
      "photoURL": "https://avatars.githubusercontent.com/u/19893222?v=4"
    }
  ],
    "createdAt": "1718826743289",
    "lastLoginAt": "1718826743290",
    "providerUsername": "Rajutsotest"
}


def test_signup():
    response = client.post("/api/v1/signup", json=signup_request_data)
    assert response.status_code in [200, 201]
    response_data = response.json()
    assert "uid" in response_data
    assert response_data["uid"] == "test-uid-123"


# Test login route using real Firebase authentication
# def test_login():
#     response = client.post("/api/v1/login", json=login_request_data)
#     assert response.status_code == 200
#     response_data = response.json()
#     assert "token" in response_data



def test_signup_existing_user():
    response = client.post("/api/v1/signup", json=signup_request_data)
    assert response.status_code == 200  
    response_data = response.json()
    assert "uid" in response_data
    assert response_data["uid"] == "test-uid-123"
