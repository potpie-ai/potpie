"""
Shared token counting for context and history management.

Uses tiktoken for consistency with OpenAI/Anthropic tokenizers. Used by
message_history_utils (prepare_multimodal_message_history) and
TokenAwareHistoryProcessor so that history trimming and compaction use the
same notion of token count.
"""

import logging
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


def get_tokenizer(model_name: Optional[str] = None) -> tiktoken.Encoding:
    """Get tiktoken encoding for the given model, or fallback to cl100k_base."""
    try:
        if model_name:
            try:
                return tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.debug(
                    "Model %s not found in tiktoken, using cl100k_base",
                    model_name,
                )
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning("Failed to get tokenizer: %s, using cl100k_base", e)
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """Count tokens in text using tiktoken.

    Uses the encoding for the given model when available; otherwise
    cl100k_base (GPT-3.5/4–style). On failure, falls back to a
    character-based estimate (~4 chars per token).

    Args:
        text: String to count.
        model_name: Optional model identifier for encoding selection.

    Returns:
        Token count (int).
    """
    if not text:
        return 0
    try:
        encoding = get_tokenizer(model_name)
        return len(encoding.encode(str(text), disallowed_special=set()))
    except Exception as e:
        logger.warning("Failed to count tokens with tiktoken: %s, using fallback", e)
        return len(str(text)) // 4
