from typing import Dict, Any, Optional

# Model configuration mappings
CHAT_MODEL_CONFIG_MAP = {
    "openai": {
        "provider": "openai",
        "low_reasoning_model": "openai/gpt-4o-mini",
        "high_reasoning_model": "openai/gpt-4o",
        "default_params": {"temperature": 0.3},
    },
    "anthropic": {
        "provider": "anthropic",
        "low_reasoning_model": "anthropic/claude-3-5-haiku-20241022",
        "high_reasoning_model": "anthropic/claude-3-7-sonnet-20250219",
        "default_params": {"temperature": 0.3},
    },
    "deepseek": {
        "provider": "deepseek",
        "low_reasoning_model": "openrouter/deepseek/deepseek-chat",
        "high_reasoning_model": "openrouter/deepseek/deepseek-chat",
        "default_params": {"temperature": 0.3},
    },
    "meta-llama": {
        "provider": "meta-llama",
        "low_reasoning_model": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "high_reasoning_model": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "default_params": {"temperature": 0.3},
    },
    "gemini": {
        "provider": "gemini",
        "low_reasoning_model": "openrouter/google/gemini-2.0-flash-001",
        "high_reasoning_model": "openrouter/google/gemini-2.0-flash-001",
        "default_params": {"temperature": 0.3},
    },
}

# Mapping for inference models
INFERENCE_MODEL_CONFIG_MAP = {
    "openai": {
        "provider": "openai",
        "low_reasoning_model": "openai/gpt-4o-mini",
        "high_reasoning_model": "openai/gpt-4o",
        "default_params": {"temperature": 0.2},
    },
    "anthropic": {
        "provider": "anthropic",
        "low_reasoning_model": "anthropic/claude-3-5-haiku-20241022",
        "high_reasoning_model": "anthropic/claude-3-7-sonnet-20250219",
        "default_params": {"temperature": 0.2},
    },
    "deepseek": {
        "provider": "deepseek",
        "low_reasoning_model": "openrouter/deepseek/deepseek-chat",
        "high_reasoning_model": "openrouter/deepseek/deepseek-chat",
        "default_params": {"temperature": 0.2},
    },
    "meta-llama": {
        "provider": "meta-llama",
        "low_reasoning_model": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "high_reasoning_model": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "default_params": {"temperature": 0.2},
    },
    "gemini": {
        "provider": "gemini",
        "low_reasoning_model": "openrouter/google/gemini-2.0-flash-001",
        "high_reasoning_model": "openrouter/google/gemini-2.0-flash-001",
        "default_params": {"temperature": 0.2},
    },
}


class LLMProviderConfig:
    def __init__(
        self,
        provider: str,
        low_reasoning_model: str,
        high_reasoning_model: str,
        default_params: Dict[str, Any],
    ):
        self.provider = provider
        self.low_reasoning_model = low_reasoning_model
        self.high_reasoning_model = high_reasoning_model
        self.default_params = default_params

    def get_model(self, size: str) -> str:
        """Get the model name based on the size."""
        return self.low_reasoning_model if size == "small" else self.high_reasoning_model

    def get_llm_params(self, size: str, api_key: str) -> Dict[str, Any]:
        """Build a complete parameter dictionary for LLM calls."""
        model_name = self.get_model(size)
        params = {
            "model": model_name,
            "temperature": self.default_params.get("temperature", 0.3),
            "api_key": api_key,
        }
        # Provider-specific tweaks can be added here
        if self.provider in ["deepseek", "anthropic"]:
            params["max_tokens"] = 8000
        return params


def build_llm_provider_config(
    user_pref: dict, config_type: str = "chat"
) -> LLMProviderConfig:
    """
    Build an LLMProviderConfig based on the user preferences.
    config_type can be 'chat' or 'inference'
    """
    if config_type == "chat":
        mapping = CHAT_MODEL_CONFIG_MAP
        selected_model = user_pref.get("chat_provider")
        default_model = "openai"
    else:
        mapping = INFERENCE_MODEL_CONFIG_MAP
        selected_model = user_pref.get("llm_provider")
        default_model = "openai"

    # Use selected model if valid, otherwise use default
    if selected_model and selected_model in mapping:
        config_data = mapping[selected_model]
    else:
        config_data = mapping[default_model]

    # Override with custom model names if specified
    low_model_key = (
        "low_reasoning_chat_model"
        if config_type == "chat"
        else "low_reasoning_inference_model"
    )
    high_model_key = (
        "high_reasoning_chat_model"
        if config_type == "chat"
        else "high_reasoning_inference_model"
    )

    low_reasoning_model = user_pref.get(low_model_key) or config_data["low_reasoning_model"]
    high_reasoning_model = user_pref.get(high_model_key) or config_data["high_reasoning_model"]

    return LLMProviderConfig(
        provider=config_data["provider"],
        low_reasoning_model=low_reasoning_model,
        high_reasoning_model=high_reasoning_model,
        default_params=config_data["default_params"],
    ) 