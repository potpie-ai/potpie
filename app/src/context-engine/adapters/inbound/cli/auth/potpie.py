"""Potpie account authentication helpers for the CLI."""

from __future__ import annotations

import json
import os
import random
import secrets
import socket
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

_DEFAULT_UI_URL = "http://localhost:3000"
_DEFAULT_API_URL = "http://localhost:8001"
_DEFAULT_PORT_MIN = 9000
_DEFAULT_PORT_MAX = 9999
_DEFAULT_TIMEOUT_SECONDS = 300.0


class PotpieCliAuthError(Exception):
    """Expected Potpie CLI authentication failure."""


@dataclass(frozen=True)
class CliCallbackResult:
    custom_token: str
    firebase_api_key: str


def resolve_potpie_ui_url() -> str:
    url = (
        os.getenv("POTPIE_UI_URL")
        or os.getenv("POTPIE_CLI_UI_BASE_URL")
        or os.getenv("POTPIE_CLI_APP_BASE_URL")
        or _DEFAULT_UI_URL
    ).strip().rstrip("/")
    if not url:
        raise PotpieCliAuthError(
            "Potpie UI URL is not set. "
            "Example: POTPIE_CLI_UI_BASE_URL=http://localhost:3000"
        )
    return url


def resolve_potpie_api_url_for_auth() -> str:
    url = (
        os.getenv("POTPIE_API_URL")
        or os.getenv("POTPIE_BASE_URL")
        or os.getenv("POTPIE_CLI_API_BASE_URL")
        or os.getenv("POTPIE_CLI_BASE_URL")
        or ""
    ).strip().rstrip("/")
    if not url:
        port = (os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT") or "").strip()
        if port:
            url = f"http://127.0.0.1:{port}"
    if not url:
        url = _DEFAULT_API_URL
    return url.rstrip("/")


def _callback_port_bounds() -> tuple[int, int]:
    lo = int(os.getenv("POTPIE_CLI_CALLBACK_PORT_MIN", str(_DEFAULT_PORT_MIN)))
    hi = int(os.getenv("POTPIE_CLI_CALLBACK_PORT_MAX", str(_DEFAULT_PORT_MAX)))
    if lo > hi:
        raise PotpieCliAuthError(
            "POTPIE_CLI_CALLBACK_PORT_MIN must be <= POTPIE_CLI_CALLBACK_PORT_MAX"
        )
    return lo, hi


def _callback_host() -> str:
    return (os.getenv("POTPIE_CLI_CALLBACK_HOST") or "localhost").strip() or "localhost"


def _auth_timeout_seconds() -> float:
    raw = (os.getenv("POTPIE_AUTH_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise PotpieCliAuthError(
            "POTPIE_AUTH_TIMEOUT_SECONDS must be a number of seconds"
        ) from exc
    if value <= 0:
        raise PotpieCliAuthError("POTPIE_AUTH_TIMEOUT_SECONDS must be positive")
    return value


def pick_callback_port() -> int:
    lo, hi = _callback_port_bounds()
    host = _callback_host()
    candidates = list(range(lo, hi + 1))
    random.shuffle(candidates)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise PotpieCliAuthError(
        f"No available port in range {lo}-{hi} on {host}. "
        "Set POTPIE_CLI_CALLBACK_PORT_MIN/MAX or free a port."
    )


def build_sign_in_url(*, ui_base_url: str, callback_url: str, state: str) -> str:
    params = urlencode({"cli_callback": callback_url, "state": state})
    return f"{ui_base_url.rstrip('/')}/sign-in?{params}"


def build_cli_success_url(*, provider: str = "potpie") -> str:
    """URL for the Potpie UI page shown after CLI auth completes."""
    slug = (provider or "potpie").strip().lower() or "potpie"
    params = urlencode({"provider": slug})
    return f"{resolve_potpie_ui_url()}/cli-success?{params}"


def open_cli_success_page(*, provider: str = "potpie") -> None:
    url = build_cli_success_url(provider=provider)
    if not webbrowser.open(url):
        print(f"Open this URL in your browser:\n{url}")


def _parse_custom_token_body(
    body: bytes,
    *,
    expected_state: str,
    header_state: str | None = None,
) -> CliCallbackResult:
    if not body:
        raise PotpieCliAuthError("CLI callback received an empty body.")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PotpieCliAuthError("CLI callback body was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise PotpieCliAuthError("CLI callback body must be a JSON object.")
    body_state = str(payload.get("state") or "").strip()
    candidate_state = str(header_state or body_state).strip()
    if not candidate_state:
        raise PotpieCliAuthError("CLI callback did not include state.")
    if not secrets.compare_digest(candidate_state, expected_state):
        raise PotpieCliAuthError("CLI callback state mismatch.")
    custom_token = str(payload.get("custom_token") or "").strip()
    if len(custom_token.split(".")) != 3:
        raise PotpieCliAuthError(
            "CLI callback did not include a valid Firebase custom token."
        )
    firebase_api_key = str(payload.get("firebase_api_key") or "").strip()
    if not firebase_api_key:
        raise PotpieCliAuthError("CLI callback did not include Firebase API config.")
    return CliCallbackResult(
        custom_token=custom_token,
        firebase_api_key=firebase_api_key,
    )


class _OneShotCallbackHandler(BaseHTTPRequestHandler):
    server_version = "PotpieCLIAuth/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args

    def _append_cors_headers(self) -> None:
        server: _OneShotCallbackServer = self.server  # type: ignore[assignment]
        self.send_header("Access-Control-Allow-Origin", server.allowed_origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Potpie-Cli-State",
        )
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "600")

    def _discard_request_body(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 0:
            self.rfile.read(length)

    def do_OPTIONS(self) -> None:  # noqa: N802
        server: _OneShotCallbackServer = self.server  # type: ignore[assignment]
        if urlparse(self.path).path != server.expected_path:
            self.send_response(404)
            self._append_cors_headers()
            self.end_headers()
            return
        self.send_response(204)
        self._append_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        server: _OneShotCallbackServer = self.server  # type: ignore[assignment]
        if urlparse(self.path).path != server.expected_path:
            self._discard_request_body()
            self.send_response(404)
            self._append_cors_headers()
            self.end_headers()
            return
        if server.received.is_set():
            self.send_response(409)
            self._append_cors_headers()
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        try:
            result = _parse_custom_token_body(
                body,
                expected_state=server.expected_state,
                header_state=self.headers.get("X-Potpie-Cli-State"),
            )
        except PotpieCliAuthError as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._append_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return
        server.result = result
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._append_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
        server.received.set()

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(405)
        self._append_cors_headers()
        self.end_headers()


class _OneShotCallbackServer(HTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        expected_state: str,
        expected_path: str,
        allowed_origin: str,
    ) -> None:
        super().__init__(server_address, _OneShotCallbackHandler)
        self.received = threading.Event()
        self.result: CliCallbackResult | None = None
        self.error: Exception | None = None
        self.expected_state = expected_state
        self.expected_path = expected_path
        self.allowed_origin = allowed_origin


def wait_for_cli_callback(
    *,
    host: str,
    port: int,
    path: str,
    state: str,
    allowed_origin: str,
    timeout_seconds: float,
) -> CliCallbackResult:
    server = _OneShotCallbackServer(
        (host, port),
        expected_state=state,
        expected_path=path if path.startswith("/") else f"/{path}",
        allowed_origin=allowed_origin,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not server.received.wait(timeout=timeout_seconds):
            raise PotpieCliAuthError(
                "Authentication timed out after "
                f"{int(timeout_seconds)} seconds. Run `potpie login` again."
            )
        if server.error is not None:
            raise server.error
        if server.result is None:
            raise PotpieCliAuthError("CLI callback did not return a custom token.")
        return server.result
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def run_browser_login_flow() -> CliCallbackResult:
    host = _callback_host()
    port = pick_callback_port()
    state = secrets.token_urlsafe(32)
    callback_path = f"/{secrets.token_urlsafe(24)}"
    callback_url = f"http://{host}:{port}{callback_path}"
    ui_url = resolve_potpie_ui_url()
    sign_in_url = build_sign_in_url(
        ui_base_url=ui_url,
        callback_url=callback_url,
        state=state,
    )
    timeout_seconds = _auth_timeout_seconds()

    opened = webbrowser.open(sign_in_url)
    if not opened:
        print(f"Open this URL in your browser:\n{sign_in_url}")

    return wait_for_cli_callback(
        host=host,
        port=port,
        path=callback_path,
        state=state,
        allowed_origin=_origin_from_url(ui_url),
        timeout_seconds=timeout_seconds,
    )


def _origin_from_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise PotpieCliAuthError("Potpie UI URL must include scheme and host.")
    return f"{parsed.scheme}://{parsed.netloc}"


def revoke_api_key_on_server(*, api_base_url: str, api_key: str) -> None:
    url = f"{api_base_url.rstrip('/')}/api/v1/api-keys"
    with httpx.Client(timeout=30.0) as client:
        response = client.delete(
            url,
            headers={"X-API-Key": api_key.strip()},
        )
    if response.status_code == 404:
        return
    if response.status_code >= 300:
        detail: Any
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise PotpieCliAuthError(f"Failed to revoke API key on server: {detail!r}")


def fetch_account_me(
    *,
    api_base_url: str,
    api_key: str | None = None,
    id_token: str | None = None,
) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/api/v1/account/me"
    headers = {"Accept": "application/json"}
    if id_token:
        headers["Authorization"] = f"Bearer {id_token.strip()}"
    elif api_key:
        headers["X-API-Key"] = api_key.strip()
    else:
        raise PotpieCliAuthError("No Potpie auth token available.")
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers)
    if response.status_code == 401:
        raise PotpieCliAuthError("Potpie session is invalid or expired.")
    if response.status_code >= 300:
        detail: Any
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise PotpieCliAuthError(f"Failed to fetch account: {detail!r}")
    data = response.json()
    return data if isinstance(data, dict) else {}
