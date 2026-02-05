"""
Central configuration for context and history management.

Phase 2: Token- and model-aware limits. Single source of truth for history
token budget, message caps, and ratios. Used by conversation_service,
prepare_multimodal_message_history, and TokenAwareHistoryProcessor.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum values to avoid breaking callers (e.g. empty history, div-by-zero)
_MIN_HISTORY_TOKEN_BUDGET = 1000
_MIN_HISTORY_MESSAGE_CAP = 1


def _env_int(key: str, default: int) -> int:
    """Parse env as int; on failure log and return default."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r, using default %s",
            key,
            raw,
            default,
        )
        return default


def _env_float(key: str, default: float) -> float:
    """Parse env as float; on failure log and return default."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r, using default %s",
            key,
            raw,
            default,
        )
        return default


# Configurable via env or config file (parsed safely to avoid import-time crashes)
HISTORY_TOKEN_BUDGET = max(
    _MIN_HISTORY_TOKEN_BUDGET,
    _env_int("HISTORY_TOKEN_BUDGET", 30000),
)
HISTORY_TOKEN_BUDGET_RATIO = max(
    0.01,
    min(1.0, _env_float("HISTORY_TOKEN_BUDGET_RATIO", 0.75)),
)
HISTORY_MESSAGE_CAP = max(
    _MIN_HISTORY_MESSAGE_CAP,
    _env_int("HISTORY_MESSAGE_CAP", 50),
)
ESTIMATED_TOKENS_PER_MESSAGE = (
    500  # for message-count proxy when tokenizer not available
)

# Protection zone: scale with context window when available
PROTECTION_ZONE_RATIO = 0.05  # last 5% of context never pruned; capped by max
PROTECTION_ZONE_MAX_TOKENS = 5000
PROTECTION_ZONE_MIN_TOKENS = 1000


def get_history_token_budget(model_string: Optional[str] = None) -> int:
    """Token budget for history; model-aware when context window is known.

    When model_string is provided and the model has a known context window,
    returns min(HISTORY_TOKEN_BUDGET, context_window * HISTORY_TOKEN_BUDGET_RATIO).
    Otherwise returns HISTORY_TOKEN_BUDGET.

    Args:
        model_string: Full model identifier (e.g. 'anthropic/claude-sonnet-4-5-20250929').
                      When None, the default budget is used.

    Returns:
        Token budget (int) to use for history.
    """
    if model_string:
        from app.modules.intelligence.provider.llm_config import get_context_window

        ctx = get_context_window(model_string)
        if ctx is not None and ctx > 0:
            budget = min(
                HISTORY_TOKEN_BUDGET,
                int(ctx * HISTORY_TOKEN_BUDGET_RATIO),
            )
            return max(_MIN_HISTORY_TOKEN_BUDGET, budget)
    return HISTORY_TOKEN_BUDGET


def get_protection_zone_tokens(context_window: Optional[int] = None) -> int:
    """Tokens in the protection zone (never pruned). Scales with context when available."""
    if context_window is not None and context_window > 0:
        zone = int(context_window * PROTECTION_ZONE_RATIO)
        zone = max(PROTECTION_ZONE_MIN_TOKENS, min(PROTECTION_ZONE_MAX_TOKENS, zone))
        return zone
    return PROTECTION_ZONE_MAX_TOKENS


# --- Phase 3: Persist compressed history per conversation ---


def _env_bool(key: str, default: bool) -> bool:
    """Parse env as bool; 'true'/'1'/'yes' -> True, else False."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes")


COMPRESSED_HISTORY_TTL_SECONDS = max(
    60,
    _env_int("COMPRESSED_HISTORY_TTL_SECONDS", 86400),
)
COMPRESSED_HISTORY_MAX_CONVERSATIONS = max(
    1,
    _env_int("COMPRESSED_HISTORY_MAX_CONVERSATIONS", 500),
)
COMPRESSED_HISTORY_STORE_BACKEND = (
    os.getenv("COMPRESSED_HISTORY_STORE_BACKEND", "memory").strip().lower()
)


def use_persisted_compressed_history() -> bool:
    """Whether to use conversation-scoped persisted compressed history (Phase 3)."""
    return _env_bool("CONTEXT_MANAGEMENT_USE_PERSISTED_COMPRESSED_HISTORY", True)


# --- Phase 4: Optional LLM summarization tier ---

CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED = _env_bool(
    "CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED", False
)
"""When True, after compaction if still over token limit, replace middle segment with LLM summary."""

CONTEXT_MANAGEMENT_SUMMARIZATION_MODEL = (
    os.getenv("CONTEXT_MANAGEMENT_SUMMARIZATION_MODEL", "").strip() or None
)
"""Optional model string for summarization (e.g. cheaper/smaller). When unset, factory uses same or cheaper model."""

SUMMARIZATION_HEAD_MESSAGES = max(
    0,
    _env_int("SUMMARIZATION_HEAD_MESSAGES", 2),
)
"""Number of messages to keep at the start (never summarized)."""

SUMMARIZATION_TAIL_MESSAGES = max(
    0,
    _env_int("SUMMARIZATION_TAIL_MESSAGES", 6),
)
"""Number of messages to keep at the end (protection zone; never summarized)."""

SUMMARIZATION_TARGET_TOKENS = max(
    500,
    _env_int("SUMMARIZATION_TARGET_TOKENS", 10_000),
)
"""Target token count for the summary that replaces the middle segment."""
