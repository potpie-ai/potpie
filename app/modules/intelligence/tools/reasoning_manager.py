"""
Reasoning Manager for tracking and saving model reasoning (TextPart) content

This module tracks TextPart content as it streams from the model and saves it
to .data/reasoning/{hash}.txt files. The hash can be referenced in diff JSON files.
"""

import hashlib
import os
from contextvars import ContextVar
from typing import Optional
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Context variable for reasoning manager - provides isolation per execution context
_reasoning_manager_ctx: ContextVar[Optional["ReasoningManager"]] = ContextVar(
    "_reasoning_manager_ctx", default=None
)


class ReasoningManager:
    """Manages reasoning content (TextPart) for a session"""

    def __init__(self):
        self.content: str = ""
        self.reasoning_hash: Optional[str] = None
        logger.debug("ReasoningManager: Created new instance")

    def append_content(self, text: str) -> None:
        """Append text content to the reasoning buffer"""
        self.content += text

    def finalize_and_save(self) -> Optional[str]:
        """
        Finalize the reasoning content, generate hash, and save to file.

        Returns:
            The reasoning hash if content was saved, None otherwise
        """
        if not self.content:
            logger.debug("ReasoningManager: No content to save")
            return None

        # Generate hash from content
        content_bytes = self.content.encode("utf-8")
        self.reasoning_hash = hashlib.sha256(content_bytes).hexdigest()

        # Save to .data/reasoning/{hash}.txt
        try:
            reasoning_dir = ".data/reasoning"
            os.makedirs(reasoning_dir, exist_ok=True)

            filepath = os.path.join(reasoning_dir, f"{self.reasoning_hash}.txt")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.content)

            logger.info(
                f"ReasoningManager: Saved reasoning content to {filepath} "
                f"(hash: {self.reasoning_hash}, size: {len(self.content)} chars)"
            )
            return self.reasoning_hash
        except Exception as e:
            logger.error(f"ReasoningManager: Failed to save reasoning content: {e}")
            return None

    def get_reasoning_hash(self) -> Optional[str]:
        """Get the current reasoning hash (None if not finalized)"""
        return self.reasoning_hash


def _get_reasoning_manager() -> ReasoningManager:
    """Get the current reasoning manager for this execution context, creating a new one if needed."""
    manager = _reasoning_manager_ctx.get()
    if manager is None:
        logger.debug(
            "ReasoningManager: Creating new manager instance for this execution context"
        )
        manager = ReasoningManager()
        _reasoning_manager_ctx.set(manager)
    return manager


def _reset_reasoning_manager() -> None:
    """Reset the reasoning manager for this execution context."""
    old_manager = _reasoning_manager_ctx.get()
    old_hash = old_manager.reasoning_hash if old_manager else None
    old_size = len(old_manager.content) if old_manager else 0
    logger.debug(
        f"ReasoningManager: Resetting manager (old hash: {old_hash}, old size: {old_size} chars)"
    )
    new_manager = ReasoningManager()
    _reasoning_manager_ctx.set(new_manager)
    logger.debug("ReasoningManager: Reset complete")
