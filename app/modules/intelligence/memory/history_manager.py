from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field

# Simple in-memory store for chat histories
store = {}

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

class InMemoryChatHistoryManager:
    """Manages chat histories in memory for different sessions."""
    @staticmethod
    def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        """Retrieve or create a new chat history for a given session."""
        if (user_id, conversation_id) not in store:
            store[(user_id, conversation_id)] = InMemoryHistory()
        return store[(user_id, conversation_id)]
