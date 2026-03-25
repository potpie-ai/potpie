"""Load PoC environment (.env.poc) and OpenRouter settings."""

from pathlib import Path

from dotenv import load_dotenv
import os

_POC_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILES = [
    _POC_ROOT / ".env",
    _POC_ROOT / ".env.poc",
]

for env_file in _ENV_FILES:
    if env_file.exists():
        load_dotenv(env_file, override=True)


def _normalize_openrouter_model_name(value: str) -> str:
    if value.startswith("openrouter/"):
        return value.removeprefix("openrouter/")
    if value.startswith("openrouter:"):
        return value.removeprefix("openrouter:")
    return value

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
MODEL_NAME = _normalize_openrouter_model_name(
    os.getenv("MODEL_NAME", "moonshotai/kimi-k2.5")
)
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "12000"))
MODEL_PARALLEL_TOOL_CALLS = (
    os.getenv("MODEL_PARALLEL_TOOL_CALLS", "true").lower() == "true"
)
MODEL_MAX_CONCURRENCY = int(os.getenv("MODEL_MAX_CONCURRENCY", "8"))

USAGE_REQUEST_LIMIT = int(os.getenv("USAGE_REQUEST_LIMIT", "150"))
USAGE_TOOL_CALLS_LIMIT = int(os.getenv("USAGE_TOOL_CALLS_LIMIT", "400"))

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "")
LOGFIRE_PROJECT_NAME = os.getenv("LOGFIRE_PROJECT_NAME", "pydantic-deep-poc")
LOGFIRE_ENVIRONMENT = os.getenv("LOGFIRE_ENVIRONMENT", os.getenv("ENV", "local"))
LOGFIRE_SEND_TO_CLOUD = os.getenv("LOGFIRE_SEND_TO_CLOUD", "true").lower() == "true"
LOGFIRE_SERVICE_NAME = os.getenv("LOGFIRE_SERVICE_NAME", "pydantic-deep-poc")
