"""Conversation memory storage."""

from typing import Optional

from potpie.runtime.storage import Storage
from potpie.types import ChatMessage


class ConversationMemory:
    """Manages conversation history storage.

    Stores messages in the relational storage backend.
    """

    def __init__(self, storage: Storage, user_id: str = "local-user"):
        """Initialize conversation memory.

        Args:
            storage: Storage backend for persistence.
            user_id: User identifier for conversation ownership.
        """
        self.storage = storage
        self.user_id = user_id

    async def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation.

        Args:
            title: Optional conversation title.

        Returns:
            Conversation ID.
        """
        # TODO: Implement conversation creation
        raise NotImplementedError(
            "ConversationMemory.create_conversation() not yet implemented"
        )

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> str:
        """Add a message to a conversation.

        Args:
            conversation_id: Conversation to add to.
            role: Message role ("user", "assistant", "system").
            content: Message content.

        Returns:
            Message ID.
        """
        # TODO: Implement message storage
        raise NotImplementedError(
            "ConversationMemory.add_message() not yet implemented"
        )

    async def get_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> list[ChatMessage]:
        """Get conversation history.

        Args:
            conversation_id: Conversation to get history for.
            limit: Maximum messages to return (most recent).

        Returns:
            List of ChatMessage objects.
        """
        # TODO: Implement history retrieval
        raise NotImplementedError(
            "ConversationMemory.get_history() not yet implemented"
        )

    async def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """List user's conversations.

        Args:
            limit: Maximum conversations to return.
            offset: Pagination offset.

        Returns:
            List of conversation metadata dicts.
        """
        # TODO: Implement conversation listing
        raise NotImplementedError(
            "ConversationMemory.list_conversations() not yet implemented"
        )

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages.

        Args:
            conversation_id: Conversation to delete.

        Returns:
            True if deleted successfully.
        """
        # TODO: Implement deletion
        raise NotImplementedError(
            "ConversationMemory.delete_conversation() not yet implemented"
        )
