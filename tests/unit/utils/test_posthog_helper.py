"""Unit tests for PostHogClient."""
import os
from unittest.mock import MagicMock, patch

import pytest

from app.modules.utils.posthog_helper import PostHogClient


pytestmark = pytest.mark.unit


class TestPostHogClientInit:
    def test_init_development_posthog_none(self):
        with patch.dict(os.environ, {"ENV": "development"}, clear=False):
            client = PostHogClient()
            assert client.posthog is None
            assert client.environment == "development"

    def test_init_production_creates_posthog(self):
        with patch.dict(
            os.environ,
            {"ENV": "production", "POSTHOG_API_KEY": "key", "POSTHOG_HOST": "https://app.posthog.com"},
            clear=False,
        ):
            with patch("app.modules.utils.posthog_helper.Posthog") as mock_posthog:
                client = PostHogClient()
                assert client.posthog is not None
                mock_posthog.assert_called_once_with("key", host="https://app.posthog.com")


class TestPostHogClientSendEvent:
    def test_send_event_not_production_returns_early(self):
        with patch.dict(os.environ, {"ENV": "development"}, clear=False):
            client = PostHogClient()
            client.send_event("user-1", "event_name", {"key": "value"})

    def test_send_event_production_capture_sync_when_no_loop(self):
        with patch.dict(
            os.environ,
            {"ENV": "production", "POSTHOG_API_KEY": "key", "POSTHOG_HOST": "https://app.posthog.com"},
            clear=False,
        ):
            with patch("app.modules.utils.posthog_helper.Posthog") as mock_posthog:
                client = PostHogClient()
                client._capture_sync = MagicMock()
                with patch("app.modules.utils.posthog_helper.asyncio.get_running_loop", side_effect=RuntimeError):
                    client.send_event("user-1", "signed_up", {})
                client._capture_sync.assert_called_once_with("user-1", "signed_up", {})


class TestPostHogClientCaptureSync:
    def test_capture_sync_posthog_none_returns(self):
        with patch.dict(os.environ, {"ENV": "development"}, clear=False):
            client = PostHogClient()
            client._capture_sync("user-1", "event", {})

    def test_capture_sync_calls_posthog_capture(self):
        with patch.dict(
            os.environ,
            {"ENV": "production", "POSTHOG_API_KEY": "key", "POSTHOG_HOST": "https://app.posthog.com"},
            clear=False,
        ):
            with patch("app.modules.utils.posthog_helper.Posthog") as mock_posthog:
                mock_instance = MagicMock()
                mock_posthog.return_value = mock_instance
                client = PostHogClient()
                client._capture_sync("user-1", "event_name", {"prop": "val"})
                mock_instance.capture.assert_called_once_with(
                    "user-1", event="event_name", properties={"prop": "val"}
                )
