"""Main Potpie facade class."""

from typing import AsyncIterator, Optional

from potpie.config import PotpieConfig
from potpie.types import ChatResponse, ParseResult


class Potpie:
    """Main facade for Potpie functionality.

    Provides high-level operations for code analysis and conversation.

    Example:
        >>> from potpie import Potpie, PotpieConfig
        >>> pp = Potpie(PotpieConfig(project_path="."))
        >>> await pp.index()
        >>> answer = await pp.ask("Where is authentication implemented?")
    """

    def __init__(self, config: Optional[PotpieConfig] = None):
        """Initialize Potpie with configuration.

        Args:
            config: Runtime configuration. Uses defaults if None.
        """
        self.config = config or PotpieConfig()
        self.config.ensure_directories()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of storage and graph backends."""
        if self._initialized:
            return
        # TODO: Initialize storage, graph_store, llm_provider
        self._initialized = True

    async def index(self, path: Optional[str] = None) -> ParseResult:
        """Index a repository, building the knowledge graph.

        Args:
            path: Path to repository. Uses config.project_path if None.

        Returns:
            ParseResult with statistics about indexed code.
        """
        await self._ensure_initialized()
        _target_path = path or self.config.project_path
        # TODO: Implement parsing and indexing
        raise NotImplementedError("index() not yet implemented")

    async def ask(self, question: str, repo_path: Optional[str] = None) -> ChatResponse:
        """Ask a one-shot question about the codebase.

        Args:
            question: Natural language question.
            repo_path: Optional path to repository context.

        Returns:
            ChatResponse with answer and source references.
        """
        await self._ensure_initialized()
        # TODO: Implement question answering
        raise NotImplementedError("ask() not yet implemented")

    async def chat(
        self, message: str, conversation_id: Optional[str] = None
    ) -> ChatResponse:
        """Send a message in a conversation context.

        Args:
            message: User message.
            conversation_id: Optional ID to continue existing conversation.

        Returns:
            ChatResponse with assistant reply.
        """
        await self._ensure_initialized()
        # TODO: Implement conversation
        raise NotImplementedError("chat() not yet implemented")

    async def chat_stream(
        self, message: str, conversation_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream a chat response.

        Args:
            message: User message.
            conversation_id: Optional ID to continue existing conversation.

        Yields:
            Chunks of the assistant's response.
        """
        await self._ensure_initialized()
        # TODO: Implement streaming
        raise NotImplementedError("chat_stream() not yet implemented")
        yield  # Make this a generator

    async def close(self) -> None:
        """Clean up resources."""
        # TODO: Close storage, graph connections
        self._initialized = False
