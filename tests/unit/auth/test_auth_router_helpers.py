"""Unit tests for auth_router helper functions."""
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from app.modules.auth.auth_router import _signup_response_with_custom_token, send_slack_message


pytestmark = pytest.mark.unit


class TestSignupResponseWithCustomToken:
    def test_no_uid_returns_unchanged(self):
        payload = {"email": "a@b.com"}
        assert _signup_response_with_custom_token(payload) == payload
        assert "customToken" not in payload

    def test_uid_no_custom_token_returns_unchanged(self):
        with patch("app.modules.auth.auth_router.AuthService.create_custom_token", return_value=None):
            payload = {"uid": "user-123", "email": "a@b.com"}
            result = _signup_response_with_custom_token(payload)
            assert result == payload
            assert "customToken" not in result

    def test_uid_with_custom_token_adds_key(self):
        with patch("app.modules.auth.auth_router.AuthService.create_custom_token", return_value="jwt-token-xyz"):
            payload = {"uid": "user-123", "email": "a@b.com"}
            result = _signup_response_with_custom_token(payload)
            assert result["customToken"] == "jwt-token-xyz"
            assert result["uid"] == "user-123"


class TestSendSlackMessage:
    @pytest.mark.asyncio
    async def test_no_webhook_url_does_nothing(self):
        with patch("app.modules.auth.auth_router.SLACK_WEBHOOK_URL", None):
            await send_slack_message("test message")

    @pytest.mark.asyncio
    async def test_with_webhook_url_posts(self):
        with patch("app.modules.auth.auth_router.SLACK_WEBHOOK_URL", "https://hooks.slack.com/x"):
            mock_post = AsyncMock()
            mock_ctx = MagicMock()
            mock_ctx.post = mock_post
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            with patch("app.modules.auth.auth_router.httpx.AsyncClient", return_value=mock_ctx):
                await send_slack_message("hello")
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"] == {"text": "hello"}
