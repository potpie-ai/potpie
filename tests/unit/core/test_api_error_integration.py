import ast
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.core.api_errors import (
    ServerErrorSanitizationMiddleware,
    logger as api_error_logger,
    register_api_error_handlers,
)
from app.core.database import get_async_db, get_db
from app.modules.auth.auth_service import AuthService
from app.modules.code_provider.code_provider_controller import CodeProviderController
from app.modules.code_provider.github.github_router import router as github_router


SENSITIVE_DATABASE_ERROR = (
    "(psycopg2.OperationalError) connection to server at "
    "pgbouncer-rw.internal.svc.cluster.local (34.118.226.109), "
    "port 5432 failed; postgresql://db-user:db-password@database.internal/app"
)


def test_main_app_registers_cors_as_outermost_user_middleware():
    main_path = Path(__file__).parents[3] / "app/main.py"
    tree = ast.parse(main_path.read_text())
    main_app = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainApp"
    )
    initializer = next(
        node
        for node in main_app.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    setup_calls = [
        call.func.attr
        for call in ast.walk(initializer)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr.startswith("setup_")
    ]

    assert setup_calls.index("setup_cors") > setup_calls.index(
        "setup_error_sanitization"
    )
    assert setup_calls.index("setup_cors") > setup_calls.index("setup_socket_io")


def test_outer_cors_adds_headers_to_sanitized_server_errors():
    app = FastAPI()
    register_api_error_handlers(app)
    app.add_middleware(ServerErrorSanitizationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://frontend.test"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/unhandled")
    async def unhandled():
        raise RuntimeError(SENSITIVE_DATABASE_ERROR)

    response = TestClient(app, raise_server_exceptions=False).get(
        "/unhandled",
        headers={"Origin": "http://frontend.test"},
    )

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "http://frontend.test"
    assert response.json()["code"] == "internal_error"


def _build_github_app() -> FastAPI:
    app = FastAPI()
    register_api_error_handlers(app)
    app.add_middleware(ServerErrorSanitizationMiddleware)
    app.include_router(github_router, prefix="/api/v1")

    def authenticated_user():
        return {"user_id": "user-123", "email": "user@example.com"}

    def database_session():
        yield object()

    async def async_database_session():
        yield None

    app.dependency_overrides[AuthService.check_auth] = authenticated_user
    app.dependency_overrides[get_db] = database_session
    app.dependency_overrides[get_async_db] = async_database_session
    return app


def test_github_user_repos_logs_database_failure_and_returns_safe_response(
    monkeypatch,
):
    async def fail_with_database_error(self, user, search=None, async_db=None):
        raise RuntimeError(SENSITIVE_DATABASE_ERROR)

    monkeypatch.setattr(
        CodeProviderController,
        "get_user_repos",
        fail_with_database_error,
    )

    records = []
    sink_id = api_error_logger.add(
        lambda message: records.append(message.record),
        level="ERROR",
    )
    try:
        response = TestClient(
            _build_github_app(),
            raise_server_exceptions=False,
        ).get(
            "/api/v1/github/user-repos",
            headers={"X-Request-ID": "request-github-repos"},
        )
    finally:
        api_error_logger.remove(sink_id)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An internal error occurred. Please try again later.",
        "code": "internal_error",
        "request_id": "request-github-repos",
    }
    assert response.headers["x-request-id"] == "request-github-repos"
    assert "pgbouncer" not in response.text.lower()
    assert "34.118.226.109" not in response.text
    assert "5432" not in response.text

    matching_records = [
        record
        for record in records
        if record["extra"].get("request_id") == "request-github-repos"
    ]
    assert matching_records
    assert any(
        "pgbouncer-rw.internal.svc.cluster.local"
        in str(record["extra"].get("private_detail", ""))
        for record in matching_records
    )
