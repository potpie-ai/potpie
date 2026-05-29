"""Unit tests for integrations_router helper functions."""
import base64
import hmac
import hashlib
import json
import time
from unittest.mock import patch

import pytest

from integrations.adapters.inbound.http.integrations_router import (
    sanitize_headers,
    truncate_content,
    get_params_summary,
    get_body_summary,
    _sign_oauth_state,
    _verify_oauth_state,
    _linear_oauth_user_message,
    _is_dev_environment,
    _resolve_oauth_state_secret,
    OAuthStateSecretMissingError,
)
import integrations.adapters.inbound.http.integrations_router as router_module


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

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    def test_no_secret_returns_raw_state(self, mock_config):
        mock_config.return_value.return_value = ""
        assert _sign_oauth_state("user-123") == "user-123"

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
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

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    def test_no_secret_returns_token_unchanged(self, mock_config):
        mock_config.return_value.return_value = ""
        assert _verify_oauth_state("user-123") == "user-123"

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    def test_invalid_token_returns_none(self, mock_config):
        mock_config.return_value.return_value = "secret"
        assert _verify_oauth_state("no-dot") is None
        assert _verify_oauth_state("bad.sig") is None

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
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

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    @patch("integrations.adapters.inbound.http.integrations_router.time")
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


class TestLinearOAuthUserMessage:
    """Verify we never surface raw exception details to the browser."""

    def test_invalid_grant_maps_to_expired_code_message(self):
        msg = _linear_oauth_user_message(Exception("invalid_grant: code already used"))
        assert "expired" in msg.lower() or "again" in msg.lower()
        assert "invalid_grant" not in msg

    def test_expired_text_maps_to_expired_code_message(self):
        msg = _linear_oauth_user_message(Exception("The code has Expired (age: 700s)"))
        assert "expired" in msg.lower() or "again" in msg.lower()
        assert "age:" not in msg

    def test_unknown_error_does_not_leak_message(self):
        # Simulate the exact shape that leaked before: psycopg2 unique violation.
        leaky = Exception(
            "(psycopg2.errors.UniqueViolation) duplicate key value violates "
            "unique constraint \"ix_integrations_unique_identifier\""
        )
        msg = _linear_oauth_user_message(leaky)
        assert "psycopg2" not in msg
        assert "UniqueViolation" not in msg
        assert "constraint" not in msg
        assert "try again" in msg.lower()

    def test_sqlalchemy_internals_not_leaked(self):
        leaky = Exception("(sqlalchemy.exc.IntegrityError) ... SQL: INSERT INTO ...")
        msg = _linear_oauth_user_message(leaky)
        assert "INSERT" not in msg
        assert "sqlalchemy" not in msg.lower()
        assert "sql" not in msg.lower()

    def test_redirect_uri_mismatch(self):
        msg = _linear_oauth_user_message(Exception("redirect_uri does not match"))
        assert "redirect" in msg.lower()


class TestOAuthStateSecretGate:
    """Verify the dev/non-dev gating on OAUTH_STATE_SECRET."""

    def setup_method(self):
        # Reset the once-per-process warning flag between tests so the
        # logger.warning expectation is reproducible.
        router_module._oauth_state_secret_warning_emitted = False

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    def test_dev_env_with_no_secret_falls_back(self, mock_config):
        mock_config.return_value.side_effect = lambda key, default="": {
            "OAUTH_STATE_SECRET": "",
            "ENV": "development",
            "ENVIRONMENT": "",
        }.get(key, default)
        assert _resolve_oauth_state_secret() is None
        # Sign returns raw state in dev fallback.
        assert _sign_oauth_state("user-1") == "user-1"
        # Verify is symmetric.
        assert _verify_oauth_state("user-1") == "user-1"

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    def test_production_with_no_secret_refuses(self, mock_config):
        mock_config.return_value.side_effect = lambda key, default="": {
            "OAUTH_STATE_SECRET": "",
            "ENV": "production",
        }.get(key, default)
        with pytest.raises(OAuthStateSecretMissingError):
            _resolve_oauth_state_secret()
        with pytest.raises(OAuthStateSecretMissingError):
            _sign_oauth_state("user-1")
        with pytest.raises(OAuthStateSecretMissingError):
            _verify_oauth_state("anything")

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    def test_staging_with_no_secret_refuses(self, mock_config):
        mock_config.return_value.side_effect = lambda key, default="": {
            "OAUTH_STATE_SECRET": "",
            "ENV": "staging",
        }.get(key, default)
        with pytest.raises(OAuthStateSecretMissingError):
            _resolve_oauth_state_secret()

    @patch("integrations.adapters.inbound.http.integrations_router.Config")
    def test_production_with_secret_signs(self, mock_config):
        mock_config.return_value.side_effect = lambda key, default="": {
            "OAUTH_STATE_SECRET": "real-secret",
            "ENV": "production",
        }.get(key, default)
        token = _sign_oauth_state("user-99")
        assert token is not None and "." in token
        assert _verify_oauth_state(token) == "user-99"

    def test_is_dev_environment_signals(self):
        with patch(
            "integrations.adapters.inbound.http.integrations_router.Config"
        ) as mock_config:
            for env in ("development", "dev", "local", "test", "testing", "DEV"):
                mock_config.return_value.side_effect = (
                    lambda k, default="", _env=env: {"ENV": _env}.get(k, default)
                )
                assert _is_dev_environment(), f"expected dev for {env!r}"
            mock_config.return_value.side_effect = lambda k, default="": {
                "ENV": "production"
            }.get(k, default)
            assert not _is_dev_environment()
