import sys
import types
import pytest
from fastapi import FastAPI, Depends
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock

# Create a fake auth_service module to avoid importing firebase/admin heavy deps
fake_auth_module = types.ModuleType("app.modules.auth.auth_service")

class FakeAuthService:
    @staticmethod
    async def check_auth():
        return {"user_id": "test-user", "email": "test@example.com"}

fake_auth_module.AuthService = FakeAuthService
sys.modules["app.modules.auth.auth_service"] = fake_auth_module

# Now import the router (this will use our fake auth module)
from app.modules.search import search_router as search_router_module
from app.core.database import get_db


@pytest.mark.asyncio
async def test_search_endpoint_rejects_empty_and_whitespace_queries():
    # Build a minimal FastAPI app and include the search router
    app = FastAPI()
    # Register the same validation handler locally so model-level validation
    # (which raises RequestValidationError -> 422) is converted to 400 for tests.
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.exception_handlers import request_validation_exception_handler

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request, exc):
        if request.url.path == "/api/v1/search":
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Search query cannot be empty or contain only whitespace"
                },
            )
        return await request_validation_exception_handler(request, exc)
    app.include_router(search_router_module.router, prefix="/api/v1")

    # Override get_db dependency to provide a dummy object (SearchService won't be invoked due to validation)
    app.dependency_overrides[get_db] = lambda: MagicMock()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Empty query
        resp = await client.post(
            "/api/v1/search",
            json={"project_id": "proj-1", "query": ""},
        )
        assert resp.status_code == 400
        assert resp.json().get("detail") == "Search query cannot be empty or contain only whitespace"

        # Whitespace-only query
        resp = await client.post(
            "/api/v1/search",
            json={"project_id": "proj-1", "query": "   \t\n "},
        )
        assert resp.status_code == 400
        assert resp.json().get("detail") == "Search query cannot be empty or contain only whitespace"
