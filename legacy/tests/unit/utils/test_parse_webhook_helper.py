"""Unit tests for ParseWebhookHelper."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.utils.parse_webhook_helper import ParseWebhookHelper


pytestmark = pytest.mark.unit


class TestParseWebhookHelperInit:
    def test_init_no_url(self):
        with patch.dict(os.environ, {}, clear=False):
            if "SLACK_PARSE_WEBHOOK_URL" in os.environ:
                del os.environ["SLACK_PARSE_WEBHOOK_URL"]
            h = ParseWebhookHelper()
            assert h.url is None

    def test_init_with_url(self):
        with patch.dict(os.environ, {"SLACK_PARSE_WEBHOOK_URL": "https://hooks.slack.com/x"}, clear=False):
            h = ParseWebhookHelper()
            assert h.url == "https://hooks.slack.com/x"


class TestParseWebhookHelperSendSlackNotification:
    @pytest.mark.asyncio
    async def test_send_no_url_does_nothing(self):
        with patch.dict(os.environ, {}, clear=False):
            h = ParseWebhookHelper()
            h.url = None
            await h.send_slack_notification("proj-1", "error message")

    @pytest.mark.asyncio
    async def test_send_with_url_posts(self):
        with patch.dict(os.environ, {"SLACK_PARSE_WEBHOOK_URL": "https://hooks.slack.com/x"}, clear=False):
            h = ParseWebhookHelper()
            mock_post = AsyncMock(return_value=MagicMock(status_code=200))
            with patch("app.modules.utils.parse_webhook_helper.httpx.AsyncClient") as mock_client:
                mock_ctx = MagicMock()
                mock_ctx.post = mock_post
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.__aexit__ = AsyncMock(return_value=None)
                mock_client.return_value = mock_ctx
                await h.send_slack_notification("proj-1", error_msg="failed")
            mock_post.assert_called_once()
            call_kw = mock_post.call_args[1]
            assert "content" in call_kw
            assert "ERROR" in call_kw["content"]
            assert "failed" in call_kw["content"]

    @pytest.mark.asyncio
    async def test_send_without_error_msg(self):
        with patch.dict(os.environ, {"SLACK_PARSE_WEBHOOK_URL": "https://hooks.slack.com/x"}, clear=False):
            h = ParseWebhookHelper()
            mock_post = AsyncMock(return_value=MagicMock(status_code=200))
            with patch("app.modules.utils.parse_webhook_helper.httpx.AsyncClient") as mock_client:
                mock_ctx = MagicMock()
                mock_ctx.post = mock_post
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.__aexit__ = AsyncMock(return_value=None)
                mock_client.return_value = mock_ctx
                await h.send_slack_notification("proj-1")
            assert "Project ID: proj-1" in mock_post.call_args[1]["content"]
