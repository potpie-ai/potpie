"""Conversation-related exceptions. Kept in a separate module to avoid circular imports with agent code."""


class GenerationCancelled(Exception):
    """Raised when generation is stopped by the user (stop API)."""

    pass
