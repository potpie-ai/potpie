import pytest


@pytest.mark.asyncio
async def test_empty_query_returns_400(client):
    payload = {"project_id": "project-id-123", "query": ""}
    resp = await client.post("/api/v1/search", json=payload)
    assert resp.status_code == 400
    assert resp.json().get("detail") == "Search query cannot be empty or contain only whitespace"


@pytest.mark.asyncio
async def test_whitespace_query_returns_400(client):
    payload = {"project_id": "project-id-123", "query": "   \t\n "}
    resp = await client.post("/api/v1/search", json=payload)
    assert resp.status_code == 400
    assert resp.json().get("detail") == "Search query cannot be empty or contain only whitespace"
