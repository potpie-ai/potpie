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


def test_request_is_https_ignores_spoofed_forwarded_proto_from_untrusted_client(
    monkeypatch,
):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
    # Spoofed forwarded proto from a non-proxy client must not flip HTTPS.
    assert not request_is_https(
        {
            "type": "http",
            "scheme": "http",
            "client": ("203.0.113.10", 1234),
            "headers": [(b"x-forwarded-proto", b"https")],
        }
    )
    assert not request_is_https(
        {
            "type": "http",
            "scheme": "http",
            "client": ("203.0.113.10", 1234),
            "headers": [(b"x-forwarded-proto", b"https, http")],
        }
    )
    assert request_is_https(
        {"type": "http", "scheme": "https", "client": ("203.0.113.10", 1), "headers": []}
    )
    assert not request_is_https(
        {"type": "http", "scheme": "http", "client": ("203.0.113.10", 1), "headers": []}
    )
    # Direct HTTPS connection still wins even if a trusted proxy sends http.
    assert request_is_https(
        {
            "type": "http",
            "scheme": "https",
            "client": ("127.0.0.1", 1),
            "headers": [(b"x-forwarded-proto", b"http")],
        }
    )


def test_request_is_https_honors_forwarded_proto_from_trusted_proxy(monkeypatch):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.5")
    assert request_is_https(
        {
            "type": "http",
            "scheme": "http",
            "client": ("10.0.0.5", 443),
            "headers": [(b"x-forwarded-proto", b"https")],
        }
    )
    assert not request_is_https(
        {
            "type": "http",
            "scheme": "http",
            "client": ("10.0.0.5", 443),
            "headers": [(b"x-forwarded-proto", b"http")],
        }
    )


def test_hsts_header_set_on_https_scheme():
    client = TestClient(_build_app(), base_url="https://testserver")
    response = client.get("/ok")

    assert response.status_code == 200
    assert response.headers["strict-transport-security"] == HSTS_HEADER_VALUE.decode(
        "latin-1"
    )


def test_hsts_header_absent_on_spoofed_forwarded_proto(monkeypatch):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
    client = TestClient(_build_app())
    response = client.get("/ok", headers={"x-forwarded-proto": "https"})

    assert response.status_code == 200
    assert "strict-transport-security" not in response.headers


def test_hsts_header_set_when_trusted_proxy_forwards_https(monkeypatch):
    # TestClient connects as host "testclient".
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "testclient")
    client = TestClient(_build_app())
    response = client.get("/ok", headers={"x-forwarded-proto": "https"})

    assert response.status_code == 200
    assert response.headers["strict-transport-security"] == HSTS_HEADER_VALUE.decode(
        "latin-1"
    )


def test_anti_framing_headers_present_on_http_and_https():
    client = TestClient(_build_app())

    http_response = client.get("/ok")
    assert http_response.headers["x-frame-options"] == X_FRAME_OPTIONS_VALUE
    assert http_response.headers["content-security-policy"] == CSP_FRAME_ANCESTORS

    https_client = TestClient(_build_app(), base_url="https://testserver")
    https_response = https_client.get("/ok")
    assert https_response.headers["x-frame-options"] == X_FRAME_OPTIONS_VALUE
    assert https_response.headers["content-security-policy"] == CSP_FRAME_ANCESTORS
    assert https_response.headers["strict-transport-security"] == HSTS_HEADER_VALUE.decode(
        "latin-1"
    )


def test_security_headers_present_on_sanitized_errors():
    client = TestClient(
        _build_app(),
        base_url="https://testserver",
        raise_server_exceptions=False,
    )
    response = client.get(
        "/boom",
        headers={
            "Origin": "http://frontend.test",
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
