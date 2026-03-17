"""Unit tests for integrations_router helper functions."""
import base64
import hmac
import hashlib
import json
import time
from unittest.mock import patch

import pytest

from app.modules.integrations.integrations_router import (
    sanitize_headers,
    truncate_content,
    get_params_summary,
    get_body_summary,
    _sign_oauth_state,
    _verify_oauth_state,
)


class TestSanitizeHeaders:
    def test_redacts_authorization(self):
        assert sanitize_headers({"Authorization": "Bearer x"})["Authorization"] == "[REDACTED]"

    def test_redacts_cookie(self):
        assert sanitize_headers({"Cookie": "session=abc"})["Cookie"] == "[REDACTED]"

    def test_redacts_signature(self):
        assert sanitize_headers({"X-Signature": "abc"})["X-Signature"] == "[REDACTED]"

    def test_preserves_safe_headers(self):
        out = sanitize_headers({"Content-Type": "application/json", "X-Request-Id": "123"})
        assert out["Content-Type"] == "application/json"
        assert out["X-Request-Id"] == "123"

    def test_empty_dict(self):
        assert sanitize_headers({}) == {}


class TestTruncateContent:
    def test_short_content_unchanged(self):
        assert truncate_content("hello", max_length=10) == "hello"

    def test_long_content_truncated(self):
        s = "a" * 300
        assert len(truncate_content(s, max_length=200)) == 203
        assert truncate_content(s, max_length=200).endswith("...")

    def test_default_max_length(self):
        assert truncate_content("x" * 250, max_length=200) == "x" * 200 + "..."


class TestGetParamsSummary:
    def test_returns_keys_and_count(self):
        params = {"a": 1, "b": "two"}
        out = get_params_summary(params)
        assert set(out["keys"]) == {"a", "b"}
        assert out["count"] == 2

    def test_preview_truncates_long_values(self):
        params = {"key": "x" * 300}
        out = get_params_summary(params)
        assert len(out["preview"]["key"]) == 203


class TestGetBodySummary:
    def test_returns_length_and_preview(self):
        body = '{"foo": "bar"}'
        out = get_body_summary(body)
        assert out["length"] == len(body)
        assert "preview" in out


class TestSignOAuthState:
    def test_none_returns_none(self):
        assert _sign_oauth_state(None) is None

    def test_empty_string_returns_none(self):
        assert _sign_oauth_state("") is None

    @patch("app.modules.integrations.integrations_router.Config")
    def test_no_secret_returns_raw_state(self, mock_config):
        mock_config.return_value.return_value = ""
        assert _sign_oauth_state("user-123") == "user-123"

    @patch("app.modules.integrations.integrations_router.Config")
    def test_with_secret_returns_signed_token(self, mock_config):
        mock_config.return_value.return_value = "my-secret"
        out = _sign_oauth_state("user-456")
        assert out is not None
        assert "." in out
        payload_b64, sig = out.rsplit(".", 1)
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        payload = json.loads(payload_json)
        assert payload["u"] == "user-456"
        assert payload["e"] > int(time.time())


class TestVerifyOAuthState:
    def test_none_returns_none(self):
        assert _verify_oauth_state(None) is None

    @patch("app.modules.integrations.integrations_router.Config")
    def test_no_secret_returns_token_unchanged(self, mock_config):
        mock_config.return_value.return_value = ""
        assert _verify_oauth_state("user-123") == "user-123"

    @patch("app.modules.integrations.integrations_router.Config")
    def test_invalid_token_returns_none(self, mock_config):
        mock_config.return_value.return_value = "secret"
        assert _verify_oauth_state("no-dot") is None
        assert _verify_oauth_state("bad.sig") is None

    @patch("app.modules.integrations.integrations_router.Config")
    def test_valid_signed_token_returns_user_id(self, mock_config):
        mock_config.return_value.return_value = "my-secret"
        expiry = int(time.time()) + 600
        payload = {"u": "user-789", "e": expiry}
        payload_json = json.dumps(payload, separators=(",", ":"))
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8")
        sig = hmac.new(
            "my-secret".encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        token = f"{payload_b64}.{sig}"
        assert _verify_oauth_state(token) == "user-789"

    @patch("app.modules.integrations.integrations_router.Config")
    @patch("app.modules.integrations.integrations_router.time")
    def test_expired_token_returns_none(self, mock_time, mock_config):
        mock_config.return_value.return_value = "my-secret"
        mock_time.time.return_value = 10000
        expiry = 9999  # already expired
        payload = {"u": "user-789", "e": expiry}
        payload_json = json.dumps(payload, separators=(",", ":"))
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8")
        sig = hmac.new(
            "my-secret".encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        token = f"{payload_b64}.{sig}"
        assert _verify_oauth_state(token) is None
