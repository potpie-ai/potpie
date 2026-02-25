import os
from typing import List, Dict, Any, Union, AsyncGenerator, Optional
from pydantic import BaseModel
from pydantic_ai.models import Model
from litellm import litellm, AsyncOpenAI, acompletion
import instructor

from app.core.config_provider import config_provider
from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient
from app.modules.utils.logger import setup_logger

from .provider_schema import (
    ProviderInfo,
    GetProviderResponse,
    AvailableModelsResponse,
    AvailableModelOption,
    SetProviderRequest,
    ModelInfo,
)
from .llm_config import (
    LLMProviderConfig,
    build_llm_provider_config,
    get_config_for_model,
    DEFAULT_CHAT_MODEL,
    DEFAULT_INFERENCE_MODEL,
)
from .exceptions import UnsupportedProviderError

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from app.modules.intelligence.provider.anthropic_caching_model import (
    CachingAnthropicModel,
)

import random
import time
import asyncio
from functools import wraps

logger = setup_logger(__name__)