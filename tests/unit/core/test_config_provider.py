"""Unit tests for ConfigProvider."""

import pytest
from unittest.mock import patch

from app.core.config_provider import ConfigProvider, MediaServiceConfigError


pytestmark = pytest.mark.unit


class TestConfigProvider:
    """Tests for ConfigProvider."""

    def test_set_neo4j_override(self):
        """set_neo4j_override stores config."""
        ConfigProvider.set_neo4j_override({"uri": "bolt://x", "username": "u", "password": "p"})
        assert ConfigProvider._neo4j_override is not None
        assert ConfigProvider._neo4j_override["uri"] == "bolt://x"
        ConfigProvider.clear_neo4j_override()

    def test_clear_neo4j_override(self):
        """clear_neo4j_override sets override to None."""
        ConfigProvider.set_neo4j_override({"uri": "x"})
        ConfigProvider.clear_neo4j_override()
        assert ConfigProvider._neo4j_override is None

    def test_get_neo4j_config_returns_override_when_set(self):
        """get_neo4j_config returns override when set."""
        override = {"uri": "bolt://override", "username": "ou", "password": "op"}
        ConfigProvider.set_neo4j_override(override)
        try:
            provider = ConfigProvider()
            cfg = provider.get_neo4j_config()
            assert cfg == override
        finally:
            ConfigProvider.clear_neo4j_override()

    def test_get_neo4j_config_returns_env_config_when_no_override(self):
        """get_neo4j_config returns env-based config when override not set."""
        ConfigProvider.clear_neo4j_override()
        with patch.dict("os.environ", {"NEO4J_URI": "bolt://local", "NEO4J_USERNAME": "n", "NEO4J_PASSWORD": "pw"}, clear=False):
            provider = ConfigProvider()
            cfg = provider.get_neo4j_config()
            assert cfg["uri"] == "bolt://local"
            assert cfg["username"] == "n"
            assert cfg["password"] == "pw"

    def test_get_github_key_replaces_literal_newlines(self):
        """get_github_key converts \\n to newline and strips."""
        with patch.dict("os.environ", {"GITHUB_PRIVATE_KEY": "line1\\nline2\\n"}, clear=False):
            provider = ConfigProvider()
            out = provider.get_github_key()
            assert out == "line1\nline2"  # .strip() removes trailing \\n -> \n

    def test_get_github_key_empty_when_not_set(self):
        """get_github_key returns falsy when not set."""
        with patch.dict("os.environ", {}, clear=False):
            if "GITHUB_PRIVATE_KEY" in __import__("os").environ:
                del __import__("os").environ["GITHUB_PRIVATE_KEY"]
            provider = ConfigProvider()
            provider.github_key = None
            assert not provider.get_github_key()

    def test_is_github_configured_true_when_both_set(self):
        """is_github_configured True when key and app id set."""
        with patch.dict("os.environ", {"GITHUB_PRIVATE_KEY": "x", "GITHUB_APP_ID": "123"}, clear=False):
            provider = ConfigProvider()
            assert provider.is_github_configured() is True

    def test_is_github_configured_false_when_missing(self):
        """is_github_configured False when key or app id missing."""
        with patch.dict("os.environ", {}, clear=False):
            provider = ConfigProvider()
            provider.github_key = None
            assert provider.is_github_configured() is False

    def test_get_demo_repo_list_returns_list(self):
        """get_demo_repo_list returns list of demo repos."""
        provider = ConfigProvider()
        repos = provider.get_demo_repo_list()
        assert isinstance(repos, list)
        assert len(repos) >= 1
        assert "id" in repos[0] and "full_name" in repos[0]

    def test_get_redis_url_default(self):
        """get_redis_url returns url without auth when no user/password."""
        with patch.dict("os.environ", {"REDISHOST": "localhost", "REDISPORT": "6379"}, clear=False):
            provider = ConfigProvider()
            url = provider.get_redis_url()
            assert "redis://" in url
            assert "localhost:6379" in url

    def test_get_redis_url_with_auth(self):
        """get_redis_url includes user and password when set."""
        with patch.dict(
            "os.environ",
            {"REDISHOST": "r", "REDISPORT": "6379", "REDISUSER": "u", "REDISPASSWORD": "p"},
            clear=False,
        ):
            provider = ConfigProvider()
            url = provider.get_redis_url()
            assert "redis://u:p@r:6379" in url or "u:p" in url

    def test_get_is_development_mode_enabled(self):
        """get_is_development_mode True when enabled."""
        with patch.dict("os.environ", {"isDevelopmentMode": "enabled"}, clear=False):
            provider = ConfigProvider()
            provider.is_development_mode = "enabled"
            assert provider.get_is_development_mode() is True

    def test_get_is_development_mode_disabled(self):
        """get_is_development_mode False when not enabled."""
        provider = ConfigProvider()
        provider.is_development_mode = "disabled"
        assert provider.get_is_development_mode() is False

    def test_get_stream_ttl_secs(self):
        """get_stream_ttl_secs returns int from env or default."""
        with patch.dict("os.environ", {"REDIS_STREAM_TTL_SECS": "1200"}, clear=False):
            assert ConfigProvider.get_stream_ttl_secs() == 1200
        with patch.dict("os.environ", {}, clear=False):
            if "REDIS_STREAM_TTL_SECS" in __import__("os").environ:
                del __import__("os").environ["REDIS_STREAM_TTL_SECS"]
            assert ConfigProvider.get_stream_ttl_secs() == 900  # default

    def test_get_stream_maxlen(self):
        """get_stream_maxlen returns int from env or default."""
        with patch.dict("os.environ", {"REDIS_STREAM_MAX_LEN": "500"}, clear=False):
            assert ConfigProvider.get_stream_maxlen() == 500

    def test_get_code_provider_type(self):
        """get_code_provider_type returns env or github."""
        with patch.dict("os.environ", {"CODE_PROVIDER": "github"}, clear=False):
            provider = ConfigProvider()
            assert provider.get_code_provider_type() == "github"

    def test_get_code_provider_token_pool(self):
        """get_code_provider_token_pool returns list from env."""
        with patch.dict("os.environ", {"CODE_PROVIDER_TOKEN_POOL": "t1, t2 , t3"}, clear=False):
            provider = ConfigProvider()
            pool = provider.get_code_provider_token_pool()
            assert pool == ["t1", "t2", "t3"]

    def test_get_is_multimodal_enabled_disabled(self):
        """get_is_multimodal_enabled False when disabled."""
        with patch.dict("os.environ", {"isMultimodalEnabled": "disabled"}, clear=False):
            provider = ConfigProvider()
            provider.is_multimodal_enabled = "disabled"
            assert provider.get_is_multimodal_enabled() is False

    def test_get_is_multimodal_enabled_enabled(self):
        """get_is_multimodal_enabled True when enabled."""
        with patch.dict("os.environ", {"isMultimodalEnabled": "enabled"}, clear=False):
            provider = ConfigProvider()
            provider.is_multimodal_enabled = "enabled"
            assert provider.get_is_multimodal_enabled() is True

    def test_get_stream_prefix(self):
        """get_stream_prefix returns env or default."""
        with patch.dict("os.environ", {"REDIS_STREAM_PREFIX": "mystream:"}, clear=False):
            assert ConfigProvider.get_stream_prefix() == "mystream:"
