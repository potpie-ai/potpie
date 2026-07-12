"""Local HTTP server that captures OAuth redirect query parameters."""

from __future__ import annotations

import html
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Sequence
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


@dataclass
class OAuthCallbackServer:
    host: str
    port: int
    path: str
    result: OAuthCallbackResult
    _done: threading.Event
    _server: HTTPServer
    _thread: threading.Thread

    def wait(self, *, timeout: float = 300.0) -> OAuthCallbackResult:
        expected_path = self.path if self.path.startswith("/") else f"/{self.path}"
        if not self._done.wait(timeout=timeout):
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for OAuth callback on "
                f"http://{self.host}:{self.port}{expected_path}"
            )
        return self.result

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def _oauth_callback_failure_html(error: str | None) -> str:
    escaped_error = html.escape(error or "No authorization code received")
    return (
        "<html><body><h1>Authentication failed</h1>"
        f"<p>{escaped_error}</p>"
        "</body></html>"
    )


def start_oauth_callback_server(
    *,
    host: str = "localhost",
    port: int,
    path: str = "/callback",
    fallback_ports: Sequence[int] = (),
) -> OAuthCallbackServer:
    """Start a local callback server, trying fallback ports when the base is busy."""
    errors: list[OSError] = []
    for candidate in _unique_ports((port, *fallback_ports)):
        try:
            return _start_oauth_callback_server(host=host, port=candidate, path=path)
        except OSError as exc:
            errors.append(exc)
    ports = ", ".join(
        str(candidate) for candidate in _unique_ports((port, *fallback_ports))
    )
    detail = "; ".join(str(exc) for exc in errors)
    raise OSError(
        f"Could not bind OAuth callback server on {host} port(s) {ports}: {detail}"
    )


def wait_for_oauth_callback(
    *,
    host: str = "localhost",
    port: int,
    path: str = "/callback",
    timeout: float = 300.0,
) -> OAuthCallbackResult:
    """Block until the provider redirects to the local callback URL."""
    server = start_oauth_callback_server(host=host, port=port, path=path)
    try:
        return server.wait(timeout=timeout)
    finally:
        server.close()


def _start_oauth_callback_server(
    *,
    host: str,
    port: int,
    path: str,
) -> OAuthCallbackServer:
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
                body = _oauth_callback_failure_html(result.error)
            self.wfile.write(body.encode("utf-8"))
            done.set()

    server = HTTPServer((host, port), _Handler)
    server.timeout = 1.0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return OAuthCallbackServer(
        host=host,
        port=port,
        path=expected_path,
        result=result,
        _done=done,
        _server=server,
        _thread=thread,
    )


def _unique_ports(ports: Sequence[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    candidates: list[int] = []
    for port in ports:
        if not 1 <= port <= 65535 or port in seen:
            continue
        seen.add(port)
        candidates.append(port)
    return tuple(candidates)


def _first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    value = values[0]
    return value if value else None
