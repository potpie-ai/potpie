from typing import Dict, Any, Optional
import os

# Default models
DEFAULT_CHAT_MODEL = "openai/gpt-5.2"
DEFAULT_INFERENCE_MODEL = "openai/gpt-5-mini"

# Default context window for unknown models (tokens)
DEFAULT_CONTEXT_WINDOW = 128000

# Model configuration mappings - now keyed by full model name
MODEL_CONFIG_MAP = {
    # OpenAI Models (GPT-5 class ~128k)
    "openai/gpt-5.2": {
        "provider": "openai",
        "context_window": 128000,
        "default_params": {"temperature": 1},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": None,
        "api_version": None,
    },
    "openai/gpt-5.1": {
        "provider": "openai",
        "context_window": 128000,
        "default_params": {"temperature": 1},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": None,
        "api_version": None,
    },
    "openai/gpt-5-mini": {
        "provider": "openai",
        "context_window": 128000,
        "default_params": {"temperature": 1},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": None,
        "api_version": None,
    },
    # Anthropic Models (Claude 3.5/4 ~200k)
    "anthropic/claude-sonnet-4-6": {
        "provider": "anthropic",
        "context_window": 200000,
        "default_params": {"temperature": 0.3, "max_tokens": 8000},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": None,
        "api_version": None,
    },
    "anthropic/claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "context_window": 200000,
        "default_params": {"temperature": 0.2, "max_tokens": 8000},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": None,
        "api_version": None,
    },
    "anthropic/claude-opus-4-6": {
        "provider": "anthropic",
        "context_window": 200000,
        "default_params": {"temperature": 0.3, "max_tokens": 8000},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": None,
        "api_version": None,
    },
    # DeepSeek Models
    "openrouter/deepseek/deepseek-v3.2-20251201": {
        "provider": "deepseek",
        "context_window": 128000,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": False,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    "openrouter/deepseek/deepseek-r1-0528": {
        "provider": "deepseek",
        "context_window": 128000,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": False,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    # Gemini Models
    "openrouter/google/gemini-2.5-pro-preview": {
        "provider": "gemini",
        "context_window": 1048576,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    "openrouter/google/gemini-3-pro-preview": {
        "provider": "gemini",
        "context_window": 1048576,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    "openrouter/google/gemini-3-flash-preview": {
        "provider": "gemini",
        "context_window": 1048576,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    "openrouter/google/gemini-3.1-pro-preview": {
        "provider": "gemini",
        "context_window": 1048576,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    # Z-AI / GLM Models
    "openrouter/z-ai/glm-5": {
        "provider": "zai",
        "context_window": 128000,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": False,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    "openrouter/z-ai/glm-4.7": {
        "provider": "zai",
        "context_window": 128000,
        "auth_provider": "openrouter",
        "default_params": {
            "temperature": 0.3,
            # Pin OpenRouter routing for GLM 4.7 to a specific backend provider
            # so we get a single provider's cost rather than an aggregate.
            # You can change this slug to any other provider from the
            # OpenRouter model page (e.g. "deepinfra", "parasail", "nebius", etc.).
            "extra_body": {
                "provider": {
                    # Example: use Nebius Token Factory as the backend
                    "order": ["nebius"],
                    "allow_fallbacks": False,
                }
            },
        },
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": False,  # Disable parallel tool calls for stability
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    # Moonshot AI / Kimi (via OpenRouter)
    "openrouter/moonshotai/kimi-k2.5": {
        "provider": "moonshot",
        "context_window": 128000,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
    # MiniMax (via OpenRouter)
    "openrouter/minimax/minimax-m2.5": {
        "provider": "minimax",
        "context_window": 196608,
        "auth_provider": "openrouter",
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": True,
            "supports_streaming": True,
            "supports_vision": False,
            "supports_tool_parallelism": True,
        },
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
    },
}


class LLMProviderConfig:
    def __init__(
        self,
        provider: str,
        model: str,
        default_params: Dict[str, Any],
        capabilities: Dict[str, bool],
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        auth_provider: Optional[str] = None,
    ):
        self.provider = provider
        self.auth_provider = auth_provider or provider
        self.model = model
        self.default_params = default_params
        self.capabilities = dict(capabilities) if capabilities else {}

        env_base_url = os.environ.get("LLM_API_BASE")
        env_api_version = os.environ.get("LLM_API_VERSION")

        self.base_url = base_url or env_base_url
        self.api_version = api_version or env_api_version

        capability_overrides = {
            "supports_pydantic": _normalize_bool_env("LLM_SUPPORTS_PYDANTIC"),
            "supports_streaming": _normalize_bool_env("LLM_SUPPORTS_STREAMING"),
            "supports_vision": _normalize_bool_env("LLM_SUPPORTS_VISION"),
            "supports_tool_parallelism": _normalize_bool_env(
                "LLM_SUPPORTS_TOOL_PARALLELISM"
            ),
        }
        for key, override in capability_overrides.items():
            if override is not None:
                self.capabilities[key] = override

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


def _normalize_bool_env(var_name: str) -> Optional[bool]:
    """Return a boolean override from the environment if present."""
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return None
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse a model string into provider and model name."""
    try:
        parts = model_string.split("/")
        provider = parts[0]

        if provider == "ollama_chat":
            provider = "ollama"
            full_model_name = "ollama/" + "/".join(parts[1:])
        else:
            full_model_name = model_string

        return provider, full_model_name
    except (IndexError, AttributeError):
        return "openai", DEFAULT_CHAT_MODEL


def get_config_for_model(model_string: str) -> Dict[str, Any]:
    """Get configuration for a specific model, with fallback to defaults."""
    if model_string in MODEL_CONFIG_MAP:
        return MODEL_CONFIG_MAP[model_string]
    # If model not found, use default configuration based on provider
    provider, _ = parse_model_string(model_string)
    env_base_url = os.environ.get("LLM_API_BASE")
    supports_pydantic = provider in {
        "openai",
        "anthropic",
        "openrouter",
        "azure",
        "ollama",
    }
    return {
        "provider": provider,
        "context_window": DEFAULT_CONTEXT_WINDOW,
        "default_params": {"temperature": 0.3},
        "capabilities": {
            "supports_pydantic": supports_pydantic or bool(env_base_url),
            "supports_streaming": True,
            "supports_vision": provider in {"openai", "anthropic"},
            "supports_tool_parallelism": provider in {"openai", "anthropic"},
        },
        "base_url": None,
        "api_version": None,
        "auth_provider": provider,
    }


def get_context_window(model_string: str) -> Optional[int]:
    """Return context window size in tokens for known models, else None.

    Uses MODEL_CONFIG_MAP and get_config_for_model (which provides a default
    for unknown models). Callers that want a value for all models can use
    the return value or fall back to DEFAULT_CONTEXT_WINDOW.

    Args:
        model_string: Full model identifier (e.g. 'anthropic/claude-sonnet-4-5-20250929').

    Returns:
        Context window size in tokens, or None if not configured (legacy entries
        without context_window will return None; get_config_for_model adds it for unknowns).
    """
    config = get_config_for_model(model_string)
    return config.get("context_window")


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
    config_data = get_config_for_model(full_model_name).copy()

    return LLMProviderConfig(
        provider=config_data["provider"],
        model=full_model_name,
        default_params=dict(config_data["default_params"]),
        capabilities=config_data.get("capabilities", {}),
        base_url=config_data.get("base_url"),
        api_version=config_data.get("api_version"),
        auth_provider=config_data.get("auth_provider"),
    )
