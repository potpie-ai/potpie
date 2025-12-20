"""Chat engine for code conversations."""

from typing import AsyncIterator, Optional

from potpie.analysis.queries import QueryEngine
from potpie.conversations.memory import ConversationMemory
from potpie.llm.provider import LLMProvider
from potpie.types import ChatMessage, ChatResponse


class ChatEngine:
    """Engine for conducting conversations about code.

    Orchestrates between LLM, query engine, and conversation memory
    to provide contextual code-aware responses.
    """

    def __init__(
        self,
        llm: LLMProvider,
        query_engine: QueryEngine,
        memory: ConversationMemory,
    ):
        """Initialize chat engine.

        Args:
            llm: LLM provider for generating responses.
            query_engine: Query engine for code context.
            memory: Conversation memory for history.
        """
        self.llm = llm
        self.query_engine = query_engine
        self.memory = memory

    async def ask(
        self,
        question: str,
        repo_path: Optional[str] = None,
    ) -> ChatResponse:
        """Ask a one-shot question about the codebase.

        Args:
            question: User's question.
            repo_path: Optional repository path for context.

        Returns:
            ChatResponse with answer and sources.
        """
        # TODO: Implement one-shot Q&A
        raise NotImplementedError("ChatEngine.ask() not yet implemented")

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
    ) -> ChatResponse:
        """Send a message in a conversation.

        Args:
            message: User's message.
            conversation_id: Optional conversation ID for context.

        Returns:
            ChatResponse with assistant reply.
        """
        # TODO: Implement conversation
        raise NotImplementedError("ChatEngine.chat() not yet implemented")

    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a chat response.

        Args:
            message: User's message.
            conversation_id: Optional conversation ID.

        Yields:
            Chunks of the response.
        """
        # TODO: Implement streaming
        raise NotImplementedError("ChatEngine.chat_stream() not yet implemented")
        yield  # Make this a generator

    async def _gather_context(self, message: str) -> list[dict]:
        """Gather relevant code context for a message.

        Args:
            message: User message to gather context for.

        Returns:
            List of context items for LLM consumption.
        """
        # TODO: Implement context gathering using query engine
        raise NotImplementedError("ChatEngine._gather_context() not yet implemented")

    def _build_prompt(
        self, message: str, context: list[dict], history: list[ChatMessage]
    ) -> list[dict[str, str]]:
        """Build LLM prompt with context and history.

        Args:
            message: Current user message.
            context: Gathered code context.
            history: Conversation history.

        Returns:
            List of messages for LLM.
        """
        # TODO: Implement prompt building
        raise NotImplementedError("ChatEngine._build_prompt() not yet implemented")
