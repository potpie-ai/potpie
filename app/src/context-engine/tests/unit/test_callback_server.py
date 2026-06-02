"""Tests for OAuth callback HTTP handler."""

from __future__ import annotations

import socket
import threading
import urllib.error
import urllib.request

import pytest

from adapters.inbound.cli.callback_server import (
    OAuthCallbackResult,
    _first,
    _oauth_callback_failure_html,
    wait_for_oauth_callback,
)


def test_oauth_callback_failure_html_escapes_error() -> None:
    body = _oauth_callback_failure_html('<script>alert("xss")</script>')
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert "alert(&quot;xss&quot;)" in body


def test_oauth_callback_failure_html_default_message() -> None:
    body = _oauth_callback_failure_html(None)
    assert "No authorization code received" in body


def test_oauth_callback_result_ok() -> None:
    ok = OAuthCallbackResult(code="abc", state="s1")
    assert ok.ok is True
    bad = OAuthCallbackResult(error="access_denied")
    assert bad.ok is False


def test_first_param_helper() -> None:
    assert _first({"code": ["abc"]}, "code") == "abc"
    assert _first({}, "code") is None
    assert _first({"code": [""]}, "code") is None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_wait_for_oauth_callback_success() -> None:
    port = _free_port()

    def hit_server() -> None:
        url = f"http://127.0.0.1:{port}/callback?code=auth-code&state=xyz"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200

    thread = threading.Thread(target=hit_server, daemon=True)
    thread.start()
    result = wait_for_oauth_callback(host="127.0.0.1", port=port, path="/callback", timeout=5.0)
    thread.join(timeout=5.0)

    assert result.ok is True
    assert result.code == "auth-code"
    assert result.state == "xyz"


def test_wait_for_oauth_callback_error_query() -> None:
    port = _free_port()

    def hit_server() -> None:
        url = f"http://127.0.0.1:{port}/callback?error=access_denied"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200

    thread = threading.Thread(target=hit_server, daemon=True)
    thread.start()
    result = wait_for_oauth_callback(host="127.0.0.1", port=port, timeout=5.0)
    thread.join(timeout=5.0)

    assert result.ok is False
    assert result.error == "access_denied"


def test_wait_for_oauth_callback_wrong_path_returns_404() -> None:
    port = _free_port()

    def hit_server() -> None:
        url = f"http://127.0.0.1:{port}/callback?code=ignored"
        try:
            urllib.request.urlopen(url, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

    thread = threading.Thread(target=hit_server, daemon=True)
    thread.start()
    with pytest.raises(TimeoutError):
        wait_for_oauth_callback(
            host="127.0.0.1",
            port=port,
            path="/other",
            timeout=0.5,
        )
    thread.join(timeout=2.0)
