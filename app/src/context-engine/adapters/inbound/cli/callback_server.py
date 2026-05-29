"""Local HTTP server that captures OAuth redirect query parameters."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class OAuthCallbackResult:
    code: str | None = None
    state: str | None = None
    error: str | None = None
    error_description: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.code) and not self.error


def wait_for_oauth_callback(
    *,
    host: str = "localhost",
    port: int,
    path: str = "/callback",
    timeout: float = 300.0,
) -> OAuthCallbackResult:
    """Block until the provider redirects to the local callback URL."""
    result = OAuthCallbackResult()
    done = threading.Event()
    expected_path = path if path.startswith("/") else f"/{path}"

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ARG002
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            params = parse_qs(parsed.query, keep_blank_values=True)
            result.code = _first(params, "code")
            result.state = _first(params, "state")
            result.error = _first(params, "error")
            result.error_description = _first(params, "error_description")

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if result.ok:
                body = (
                    "<html><body><h1>Authentication successful</h1>"
                    "<p>You can close this tab and return to the terminal.</p>"
                    "</body></html>"
                )
            else:
                body = (
                    "<html><body><h1>Authentication failed</h1>"
                    f"<p>{result.error or 'No authorization code received'}</p>"
                    "</body></html>"
                )
            self.wfile.write(body.encode("utf-8"))
            done.set()

    server = HTTPServer((host, port), _Handler)
    server.timeout = 1.0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not done.wait(timeout=timeout):
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for OAuth callback on "
                f"http://{host}:{port}{expected_path}"
            )
        return result
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    value = values[0]
    return value if value else None
