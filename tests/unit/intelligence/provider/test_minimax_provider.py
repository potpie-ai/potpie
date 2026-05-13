"""Unit tests for MiniMax direct provider support.

Tests cover:
- MODEL_CONFIG_MAP entries for direct MiniMax models
- AVAILABLE_MODELS entries for direct MiniMax models
- LLMProviderConfig construction from MiniMax model config
- parse_model_string and get_config_for_model for minimax/ prefix
- minimax in openai_like_providers for pydantic-ai model routing
- API key resolution via MINIMAX_API_KEY env var
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from app.modules.intelligence.provider.llm_config import (
    MODEL_CONFIG_MAP,
    LLMProviderConfig,
    parse_model_string,
    get_config_for_model,
    build_llm_provider_config,
)
from app.modules.intelligence.provider.provider_service import (
    AVAILABLE_MODELS,
    PLATFORM_PROVIDERS,
    ProviderService,
)


# ── MODEL_CONFIG_MAP ──────────────────────────────────────────────


class TestMiniMaxModelConfig:
    """Verify direct MiniMax entries in MODEL_CONFIG_MAP."""

    DIRECT_MODELS = [
        "minimax/MiniMax-M2.7",
        "minimax/MiniMax-M2.7-highspeed",
        "minimax/MiniMax-M2.5",
        "minimax/MiniMax-M2.5-highspeed",
    ]

    @pytest.mark.parametrize("model_id", DIRECT_MODELS)
    def test_model_in_config_map(self, model_id):
        assert model_id in MODEL_CONFIG_MAP, f"{model_id} missing from MODEL_CONFIG_MAP"

    @pytest.mark.parametrize("model_id", DIRECT_MODELS)
    def test_provider_is_minimax(self, model_id):
        assert MODEL_CONFIG_MAP[model_id]["provider"] == "minimax"

    @pytest.mark.parametrize("model_id", DIRECT_MODELS)
    def test_base_url_is_minimax_api(self, model_id):
        assert MODEL_CONFIG_MAP[model_id]["base_url"] == "https://api.minimax.io/v1"

    @pytest.mark.parametrize("model_id", DIRECT_MODELS)
    def test_capabilities(self, model_id):
        caps = MODEL_CONFIG_MAP[model_id]["capabilities"]
        assert caps["supports_pydantic"] is True
        assert caps["supports_streaming"] is True
        assert caps["supports_vision"] is False
        assert caps["supports_tool_parallelism"] is True

    def test_m27_context_window(self):
        assert MODEL_CONFIG_MAP["minimax/MiniMax-M2.7"]["context_window"] == 1_000_000

    def test_m27_highspeed_context_window(self):
        assert MODEL_CONFIG_MAP["minimax/MiniMax-M2.7-highspeed"]["context_window"] == 1_000_000

    def test_m25_context_window(self):
        assert MODEL_CONFIG_MAP["minimax/MiniMax-M2.5"]["context_window"] == 200_000

    def test_m25_highspeed_context_window(self):
        assert MODEL_CONFIG_MAP["minimax/MiniMax-M2.5-highspeed"]["context_window"] == 204_000

    def test_openrouter_entry_still_present(self):
        """The existing OpenRouter-proxied entry must not be removed."""
        assert "openrouter/minimax/minimax-m2.5" in MODEL_CONFIG_MAP

    def test_no_auth_provider_override_for_direct(self):
        """Direct entries should not have an auth_provider (defaults to provider)."""
        for model_id in self.DIRECT_MODELS:
            assert "auth_provider" not in MODEL_CONFIG_MAP[model_id] or \
                   MODEL_CONFIG_MAP[model_id].get("auth_provider") is None


# ── parse_model_string / get_config_for_model ─────────────────────


class TestModelStringParsing:
    def test_parse_minimax_model(self):
        provider, full = parse_model_string("minimax/MiniMax-M2.7")
        assert provider == "minimax"
        assert full == "minimax/MiniMax-M2.7"

    def test_get_config_for_known_minimax_model(self):
        config = get_config_for_model("minimax/MiniMax-M2.7")
        assert config["provider"] == "minimax"
        assert config["base_url"] == "https://api.minimax.io/v1"

    def test_get_config_fallback_for_unknown_minimax_model(self):
        """Unknown minimax/* model still returns a config with supports_pydantic."""
        config = get_config_for_model("minimax/some-future-model")
        assert config["capabilities"]["supports_pydantic"] is True


# ── LLMProviderConfig ──────────────────────────────────────────────


class TestLLMProviderConfigMiniMax:
    def test_build_from_minimax_config(self):
        config_data = MODEL_CONFIG_MAP["minimax/MiniMax-M2.7"]
        cfg = LLMProviderConfig(
            provider=config_data["provider"],
            model="minimax/MiniMax-M2.7",
            default_params=dict(config_data["default_params"]),
            capabilities=config_data["capabilities"],
            base_url=config_data["base_url"],
        )
        assert cfg.provider == "minimax"
        assert cfg.base_url == "https://api.minimax.io/v1"
        assert cfg.capabilities["supports_pydantic"] is True

    def test_get_llm_params_includes_model(self):
        config_data = MODEL_CONFIG_MAP["minimax/MiniMax-M2.7"]
        cfg = LLMProviderConfig(
            provider=config_data["provider"],
            model="minimax/MiniMax-M2.7",
            default_params=dict(config_data["default_params"]),
            capabilities=config_data["capabilities"],
            base_url=config_data["base_url"],
        )
        params = cfg.get_llm_params("test-key")
        assert params["model"] == "minimax/MiniMax-M2.7"
        assert params["api_key"] == "test-key"
        assert params["temperature"] == 0.3

    @patch.dict(os.environ, {"CHAT_MODEL": "minimax/MiniMax-M2.7"}, clear=False)
    def test_build_llm_provider_config_from_env(self):
        cfg = build_llm_provider_config({}, config_type="chat")
        assert cfg.provider == "minimax"
        assert cfg.model == "minimax/MiniMax-M2.7"
        assert cfg.base_url == "https://api.minimax.io/v1"


# ── AVAILABLE_MODELS ──────────────────────────────────────────────


class TestAvailableModels:
    DIRECT_IDS = [
        "minimax/MiniMax-M2.7",
        "minimax/MiniMax-M2.7-highspeed",
        "minimax/MiniMax-M2.5",
        "minimax/MiniMax-M2.5-highspeed",
    ]

    @pytest.mark.parametrize("model_id", DIRECT_IDS)
    def test_direct_model_in_available(self, model_id):
        ids = [m.id for m in AVAILABLE_MODELS]
        assert model_id in ids, f"{model_id} missing from AVAILABLE_MODELS"

    @pytest.mark.parametrize("model_id", DIRECT_IDS)
    def test_provider_is_minimax(self, model_id):
        model = next(m for m in AVAILABLE_MODELS if m.id == model_id)
        assert model.provider == "minimax"

    @pytest.mark.parametrize("model_id", DIRECT_IDS)
    def test_is_chat_and_inference(self, model_id):
        model = next(m for m in AVAILABLE_MODELS if m.id == model_id)
        assert model.is_chat_model is True
        assert model.is_inference_model is True

    def test_openrouter_entry_still_available(self):
        ids = [m.id for m in AVAILABLE_MODELS]
        assert "openrouter/minimax/minimax-m2.5" in ids

    def test_minimax_in_platform_providers(self):
        assert "minimax" in PLATFORM_PROVIDERS


# ── ProviderService.get_pydantic_model routing ────────────────────


class TestPydanticModelRouting:
    """Verify minimax is treated as an OpenAI-like provider in get_pydantic_model."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"}, clear=False)
    def test_minimax_routes_to_openai_model(self):
        """When MINIMAX_API_KEY is set, get_pydantic_model returns an OpenAIModel."""
        mock_db = MagicMock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        service = ProviderService(mock_db, "test-user")
        model = service.get_pydantic_model(model="minimax/MiniMax-M2.7")

        from pydantic_ai.models.openai import OpenAIModel
        assert isinstance(model, OpenAIModel)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"}, clear=False)
    def test_minimax_highspeed_routes_to_openai_model(self):
        mock_db = MagicMock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        service = ProviderService(mock_db, "test-user")
        model = service.get_pydantic_model(model="minimax/MiniMax-M2.7-highspeed")

        from pydantic_ai.models.openai import OpenAIModel
        assert isinstance(model, OpenAIModel)


# ── API key resolution ────────────────────────────────────────────


class TestAPIKeyResolution:
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mk-test-123"}, clear=False)
    def test_minimax_api_key_from_env(self):
        mock_db = MagicMock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        service = ProviderService(mock_db, "test-user")
        key = service._get_api_key("minimax")
        assert key == "mk-test-123"

    @patch.dict(os.environ, {}, clear=False)
    def test_falls_back_to_llm_api_key(self):
        # Remove MINIMAX_API_KEY if present
        os.environ.pop("MINIMAX_API_KEY", None)
        os.environ["LLM_API_KEY"] = "fallback-key"
        try:
            mock_db = MagicMock()
            mock_db.query.return_value.filter_by.return_value.first.return_value = None

            service = ProviderService(mock_db, "test-user")
            key = service._get_api_key("minimax")
            assert key == "fallback-key"
        finally:
            os.environ.pop("LLM_API_KEY", None)


# ── Temperature default ──────────────────────────────────────────


class TestTemperatureDefaults:
    @pytest.mark.parametrize("model_id", [
        "minimax/MiniMax-M2.7",
        "minimax/MiniMax-M2.7-highspeed",
        "minimax/MiniMax-M2.5",
        "minimax/MiniMax-M2.5-highspeed",
    ])
    def test_default_temperature(self, model_id):
        assert MODEL_CONFIG_MAP[model_id]["default_params"]["temperature"] == 0.3
