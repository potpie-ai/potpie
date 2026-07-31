from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.core.api_errors import (
    ServerErrorSanitizationMiddleware,
    register_api_error_handlers,
)
from app.core.security_headers import (
    CSP_FRAME_ANCESTORS,
    HSTS_HEADER_VALUE,
    SecurityHeadersMiddleware,
    X_FRAME_OPTIONS_VALUE,
    request_is_https,
)


SENSITIVE_ERROR = (
    "connection to server at pgbouncer-rw.internal.svc.cluster.local "
    "(34.118.226.109), port 5432 failed"
)


def _build_app() -> FastAPI:
    app = FastAPI()
    register_api_error_handlers(app)
    app.add_middleware(ServerErrorSanitizationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://frontend.test"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/boom")
    async def boom():
        raise RuntimeError(SENSITIVE_ERROR)

    return app


def test_request_is_https_prefers_forwarded_proto():
    assert request_is_https(
        {
            "type": "http",
            "scheme": "http",
            "headers": [(b"x-forwarded-proto", b"https")],
        }
    )
    assert request_is_https(
        {
            "type": "http",
            "scheme": "http",
            "headers": [(b"x-forwarded-proto", b"https, http")],
        }
    )
    assert not request_is_https(
        {
            "type": "http",
            "scheme": "http",
            "headers": [(b"x-forwarded-proto", b"http")],
        }
    )
    assert request_is_https({"type": "http", "scheme": "https", "headers": []})
    assert not request_is_https({"type": "http", "scheme": "http", "headers": []})


def test_hsts_header_set_on_https_forwarded_requests():
    client = TestClient(_build_app())
    response = client.get("/ok", headers={"x-forwarded-proto": "https"})

    assert response.status_code == 200
    assert response.headers["strict-transport-security"] == HSTS_HEADER_VALUE.decode(
        "latin-1"
    )


def test_hsts_header_absent_on_plain_http():
    client = TestClient(_build_app())
    response = client.get("/ok")

    assert response.status_code == 200
    assert "strict-transport-security" not in response.headers


def test_anti_framing_headers_present_on_http_and_https():
    client = TestClient(_build_app())

    http_response = client.get("/ok")
    assert http_response.headers["x-frame-options"] == X_FRAME_OPTIONS_VALUE
    assert http_response.headers["content-security-policy"] == CSP_FRAME_ANCESTORS

    https_response = client.get("/ok", headers={"x-forwarded-proto": "https"})
    assert https_response.headers["x-frame-options"] == X_FRAME_OPTIONS_VALUE
    assert https_response.headers["content-security-policy"] == CSP_FRAME_ANCESTORS
    assert https_response.headers["strict-transport-security"] == HSTS_HEADER_VALUE.decode(
        "latin-1"
    )


def test_security_headers_present_on_sanitized_errors():
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get(
        "/boom",
        headers={
            "Origin": "http://frontend.test",
            "x-forwarded-proto": "https",
        },
    )

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert response.headers["access-control-allow-origin"] == "http://frontend.test"
    assert response.headers["strict-transport-security"] == HSTS_HEADER_VALUE.decode(
        "latin-1"
    )
    assert response.headers["x-frame-options"] == X_FRAME_OPTIONS_VALUE
    assert response.headers["content-security-policy"] == CSP_FRAME_ANCESTORS
