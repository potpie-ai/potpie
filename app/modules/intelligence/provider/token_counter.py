"""Token counting service for context window management."""
import logging
from typing import Dict, List, Optional
import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Manages token counting for different model types."""

    def __init__(self):
        # Cache encoders to avoid repeated initialization
        self._encoders: Dict[str, tiktoken.Encoding] = {}

    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create tiktoken encoder for model."""
        # Map model strings to tiktoken encoding names
        encoding_map = {
            # OpenAI models
            "gpt-4": "cl100k_base",
            "gpt-4o": "o200k_base",
            "gpt-4.1": "o200k_base",
            "o4": "o200k_base",
            # Anthropic (use cl100k_base as approximation)
            "claude": "cl100k_base",
            # Default fallback
            "default": "cl100k_base",
        }

        # Determine encoding name
        encoding_name = "default"
        model_lower = model.lower()
        for key, enc_name in encoding_map.items():
            if key in model_lower:
                encoding_name = enc_name
                break

        # Cache encoder
        if encoding_name not in self._encoders:
            try:
                actual_encoding = encoding_map.get(encoding_name, "cl100k_base")
                self._encoders[encoding_name] = tiktoken.get_encoding(actual_encoding)
            except Exception as e:
                logger.warning(f"Failed to get encoding {encoding_name}: {e}, using cl100k_base")
                self._encoders[encoding_name] = tiktoken.get_encoding("cl100k_base")

        return self._encoders[encoding_name]

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for given model."""
        if not text:
            return 0

        try:
            encoder = self._get_encoder(model)
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimate of 4 chars per token
            return len(text) // 4

    def count_messages_tokens(
        self,
        messages: List[str],
        model: str,
        tokens_per_message: int = 3,
        tokens_per_name: int = 1
    ) -> int:
        """
        Count tokens in a list of message strings.
        Adds overhead tokens for message formatting.
        """
        total_tokens = 0

        for message in messages:
            total_tokens += tokens_per_message
            total_tokens += self.count_tokens(message, model)

        # Add tokens for reply priming
        total_tokens += 3

        return total_tokens

    def get_context_limit(self, model: str) -> int:
        """
        Get context window size for a model.
        This will query the LLMProviderConfig.
        """
        from app.modules.intelligence.provider.llm_config import get_config_for_model

        config = get_config_for_model(model)
        return config.get("context_window", 8192)

    def estimate_file_tokens(self, file_size_bytes: int, mime_type: str) -> int:
        """
        Estimate tokens from file size before extraction.

        Heuristics:
        - Text files: ~4 characters per token
        - PDFs: ~5 characters per token (includes formatting overhead)
        - DOCX: ~5 characters per token
        - CSV: ~6 characters per token (structured data)
        """
        if "pdf" in mime_type:
            return file_size_bytes // 5
        elif "word" in mime_type or "document" in mime_type:
            return file_size_bytes // 5
        elif "csv" in mime_type or "spreadsheet" in mime_type:
            return file_size_bytes // 6
        else:  # text/plain, markdown, code
            return file_size_bytes // 4

    def calculate_context_usage(
        self,
        conversation_history: List[str],
        additional_context: str,
        attachment_tokens: int,
        model: str
    ) -> Dict[str, int]:
        """Calculate total context usage breakdown."""
        history_tokens = self.count_messages_tokens(conversation_history, model)
        context_tokens = self.count_tokens(additional_context, model)

        return {
            "conversation_history": history_tokens,
            "additional_context": context_tokens,
            "attachments": attachment_tokens,
            "total": history_tokens + context_tokens + attachment_tokens,
            "model_limit": self.get_context_limit(model),
        }


# Global singleton instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get or create global TokenCounter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter
