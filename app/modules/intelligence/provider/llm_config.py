from typing import Dict, Any
import os

# Default models
DEFAULT_CHAT_MODEL = "openai/gpt-4o"
DEFAULT_INFERENCE_MODEL = "openai/gpt-4.1-mini"

# Model configuration mappings - now keyed by full model name
MODEL_CONFIG_MAP = {
    # OpenAI Models
    "openai/gpt-4.1-mini": {
        "provider": "openai",
        "default_params": {"temperature": 0.3},
    },
    "openai/gpt-4.1": {
        "provider": "openai",
        "default_params": {"temperature": 0.3},
    },
    "openai/gpt-4o": {
        "provider": "openai",
        "default_params": {"temperature": 0.3},
    },
    # Anthropic Models
    "anthropic/claude-3-7-sonnet-20250219": {
        "provider": "anthropic",
        "default_params": {"temperature": 0.3, "max_tokens": 8000},
    },
    "anthropic/claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "default_params": {"temperature": 0.2, "max_tokens": 8000},
    },
    # DeepSeek Models
    "openrouter/deepseek/deepseek-chat-v3-0324": {
        "provider": "deepseek",
        "default_params": {"temperature": 0.3, "max_tokens": 8000},
    },
    # Meta-Llama Models
    "openrouter/meta-llama/llama-3.3-70b-instruct": {
        "provider": "meta-llama",
        "default_params": {"temperature": 0.3},
    },
    # Gemini Models
    "openrouter/google/gemini-2.0-flash-001": {
        "provider": "gemini",
        "default_params": {"temperature": 0.3},
    },
}


class LLMProviderConfig:
    def __init__(
        self,
        provider: str,
        model: str,
        default_params: Dict[str, Any],
    ):
        self.provider = provider
        self.model = model
        self.default_params = default_params

    def get_llm_params(self, api_key: str) -> Dict[str, Any]:
        """Build a complete parameter dictionary for LLM calls."""
        params = {
            "model": self.model,
            "temperature": self.default_params.get("temperature", 0.3),
            "api_key": api_key,
        }
        # Add any additional default parameters
        for key, value in self.default_params.items():
            if key != "temperature":  # temperature already handled above
                params[key] = value
        return params


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse a model string into provider and model name."""
    try:
        provider = model_string.split("/")[0]
        return provider, model_string
    except (IndexError, AttributeError):
        return "openai", DEFAULT_CHAT_MODEL


def get_config_for_model(model_string: str) -> Dict[str, Any]:
    """Get configuration for a specific model, with fallback to defaults."""
    if model_string in MODEL_CONFIG_MAP:
        return MODEL_CONFIG_MAP[model_string]
    # If model not found, use default configuration based on provider
    provider, _ = parse_model_string(model_string)
    return {
        "provider": provider,
        "default_params": {"temperature": 0.3},
    }


def build_llm_provider_config(
    user_pref: dict, config_type: str = "chat"
) -> LLMProviderConfig:
    """
    Build an LLMProviderConfig based on the environment variables, user preferences, and defaults.
    Config type can be 'chat' or 'inference'.

    Priority order:
    1. Environment variables (CHAT_MODEL or INFERENCE_MODEL)
    2. User preferences (chat_model or inference_model)
    3. Built-in defaults
    """
    # Determine which model to use based on config_type and priority order
    if config_type == "chat":
        model_string = (
            os.environ.get("CHAT_MODEL")
            or user_pref.get("chat_model")
            or DEFAULT_CHAT_MODEL
        )
    else:
        model_string = (
            os.environ.get("INFERENCE_MODEL")
            or user_pref.get("inference_model")
            or DEFAULT_INFERENCE_MODEL
        )

    # Get provider and configuration for the model
    provider, full_model_name = parse_model_string(model_string)
    config_data = get_config_for_model(full_model_name)

    return LLMProviderConfig(
        provider=config_data["provider"],
        model=full_model_name,
        default_params=config_data["default_params"],
    )
