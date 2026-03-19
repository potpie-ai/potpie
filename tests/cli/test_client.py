"""Tests for the PotpieClient HTTP client."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from potpie.cli.client import PotpieClient, API_PREFIX


class TestPotpieClientInit:
    """Test client initialization."""

    def test_default_base_url(self):
        client = PotpieClient()
        assert client.base_url == "http://localhost:8001"

    def test_custom_base_url(self):
        client = PotpieClient(base_url="http://myhost:9000")
        assert client.base_url == "http://myhost:9000"

    def test_strips_trailing_slash(self):
        client = PotpieClient(base_url="http://localhost:8001/")
        assert client.base_url == "http://localhost:8001"

    def test_explicit_api_key(self):
        client = PotpieClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_explicit_user_id(self):
        client = PotpieClient(user_id="test-user")
        assert client._user_id == "test-user"


class TestHeaders:
    """Test authentication header generation."""

    def test_headers_include_api_key(self):
        client = PotpieClient(api_key="my-key", user_id="my-user")
        headers = client._headers()
        assert headers["x-api-key"] == "my-key"
        assert headers["x-user-id"] == "my-user"

    def test_headers_empty_api_key(self):
        client = PotpieClient(api_key="", user_id="")
        headers = client._headers()
        assert "x-api-key" not in headers


class TestIsAlive:
    """Test is_alive health checks."""

    @patch.object(PotpieClient, "health", return_value={"status": "ok"})
    def test_returns_true_when_healthy(self, mock_health):
        client = PotpieClient()
        assert client.is_alive() is True

    @patch.object(PotpieClient, "health", side_effect=Exception("Connection refused"))
    def test_returns_false_on_error(self, mock_health):
        client = PotpieClient()
        assert client.is_alive() is False


class TestPollParsing:
    """Test polling logic."""

    @patch.object(PotpieClient, "get_parsing_status")
    def test_returns_on_ready(self, mock_status):
        mock_status.return_value = {"status": "ready"}
        client = PotpieClient()
        result = client.poll_parsing("test-id", interval=0.01)
        assert result["status"] == "ready"

    @patch.object(PotpieClient, "get_parsing_status")
    def test_returns_on_error(self, mock_status):
        mock_status.return_value = {"status": "error", "message": "failed"}
        client = PotpieClient()
        result = client.poll_parsing("test-id", interval=0.01)
        assert result["status"] == "error"

    @patch.object(PotpieClient, "get_parsing_status")
    def test_timeout(self, mock_status):
        mock_status.return_value = {"status": "processing"}
        client = PotpieClient()
        with pytest.raises(TimeoutError):
            client.poll_parsing("test-id", interval=0.01, max_wait=0.05)


class TestAPIPrefix:
    """Test that API prefix is correctly configured."""

    def test_api_prefix_value(self):
        assert API_PREFIX == "/api/v1"
