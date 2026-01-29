"""Main PotpieRuntime class - entry point for the library."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from potpie.config import RuntimeConfig, RuntimeConfigBuilder
from potpie.core.database import DatabaseManager
from potpie.core.neo4j import Neo4jManager
from potpie.core.redis import RedisManager
from potpie.exceptions import NotInitializedError, PotpieError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from potpie.resources.projects import ProjectResource
    from potpie.resources.parsing import ParsingResource
    from potpie.resources.users import UserResource
    from potpie.agents.runner import AgentRunner

logger = logging.getLogger(__name__)


class PotpieRuntime:
    """Main entry point for the PotpieRuntime library.

    Manages lifecycle of all core components and provides access to
    resources and agents. This is a user-agnostic admin layer - user
    context is passed per-operation, not at construction time.

    Usage:
        # Option 1: Async context manager (recommended)
        async with PotpieRuntime(config) as runtime:
            project = await runtime.projects.get(project_id)

        # Option 2: Manual lifecycle
        runtime = PotpieRuntime(config)
        await runtime.initialize()
        try:
            project = await runtime.projects.get(project_id)
        finally:
            await runtime.close()

        # Option 3: From environment
        runtime = PotpieRuntime.from_env()

        # Option 4: Builder pattern
        runtime = (
            PotpieRuntime.builder()
            .postgres("postgresql://...")
            .neo4j("bolt://...", "neo4j", "password")
            .build_runtime()
        )
    """

    def __init__(self, config: RuntimeConfig):
        """Initialize PotpieRuntime with configuration.

        Args:
            config: Runtime configuration
        """
        config.validate()
        self._config = config
        self._db_manager: Optional[DatabaseManager] = None
        self._neo4j_manager: Optional[Neo4jManager] = None
        self._redis_manager: Optional[RedisManager] = None
        self._initialized = False

        # Lazy-initialized resource accessors (Phase 2)
        self._projects: Optional[ProjectResource] = None
        self._parsing: Optional[ParsingResource] = None
        self._users: Optional[UserResource] = None
        self._conversations = None

        # Lazy-initialized agent runner (Phase 3)
        self._agents: Optional[AgentRunner] = None
        self._agent_session: Optional[Session] = None

    @classmethod
    def from_env(
        cls,
        env_prefix: str = "",
        dotenv_path: Optional[str] = None,
    ) -> PotpieRuntime:
        """Create PotpieRuntime from environment variables.

        Args:
            env_prefix: Optional prefix for environment variables
            dotenv_path: Optional path to .env file to load

        Returns:
            Configured PotpieRuntime instance (not yet initialized)
        """
        if dotenv_path:
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path, override=True)
            except ImportError:
                logger.warning("python-dotenv not installed, .env file not loaded")

        config = RuntimeConfig.from_env(env_prefix)
        return cls(config)

    @classmethod
    def builder(cls) -> RuntimeConfigBuilder:
        """Get a builder for creating RuntimeConfig.

        Returns:
            RuntimeConfigBuilder for fluent configuration

        Example:
            runtime = (
                PotpieRuntime.builder()
                .postgres("postgresql://...")
                .neo4j("bolt://...", "neo4j", "password")
                .llm("openai", api_key="sk-...")
                .build_runtime()
            )
        """
        return _RuntimeBuilder()

    @property
    def config(self) -> RuntimeConfig:
        """Get the runtime configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the runtime has been initialized."""
        return self._initialized

    @property
    def db(self) -> DatabaseManager:
        """Get the database manager.

        Raises:
            NotInitializedError: If runtime not initialized
        """
        if not self._initialized or self._db_manager is None:
            raise NotInitializedError(
                "Runtime not initialized - call initialize() first"
            )
        return self._db_manager

    @property
    def neo4j(self) -> Neo4jManager:
        """Get the Neo4j manager.

        Raises:
            NotInitializedError: If runtime not initialized
        """
        if not self._initialized or self._neo4j_manager is None:
            raise NotInitializedError(
                "Runtime not initialized - call initialize() first"
            )
        return self._neo4j_manager

    @property
    def redis(self) -> RedisManager:
        """Get the Redis manager.

        Raises:
            NotInitializedError: If runtime not initialized
        """
        if not self._initialized or self._redis_manager is None:
            raise NotInitializedError(
                "Runtime not initialized - call initialize() first"
            )
        return self._redis_manager

    @property
    def projects(self) -> ProjectResource:
        """Access project resources.

        Returns:
            ProjectResource for managing projects

        Raises:
            NotInitializedError: If runtime not initialized
        """
        if not self._initialized:
            raise NotInitializedError(
                "Runtime not initialized - call initialize() first"
            )

        if self._projects is None:
            from potpie.resources.projects import ProjectResource

            self._projects = ProjectResource(
                config=self._config,
                db_manager=self._db_manager,
                neo4j_manager=self._neo4j_manager,
            )
        return self._projects

    @property
    def parsing(self) -> ParsingResource:
        """Access parsing resources.

        Returns:
            ParsingResource for parsing projects

        Raises:
            NotInitializedError: If runtime not initialized
        """
        if not self._initialized:
            raise NotInitializedError(
                "Runtime not initialized - call initialize() first"
            )

        if self._parsing is None:
            from potpie.resources.parsing import ParsingResource

            self._parsing = ParsingResource(
                config=self._config,
                db_manager=self._db_manager,
                neo4j_manager=self._neo4j_manager,
            )
        return self._parsing

    @property
    def users(self) -> UserResource:
        """Access user resources.

        Returns:
            UserResource for managing users

        Raises:
            NotInitializedError: If runtime not initialized
        """
        if not self._initialized:
            raise NotInitializedError(
                "Runtime not initialized - call initialize() first"
            )

        if self._users is None:
            from potpie.resources.users import UserResource

            self._users = UserResource(
                config=self._config,
                db_manager=self._db_manager,
                neo4j_manager=self._neo4j_manager,
            )
        return self._users

    @property
    def agents(self) -> AgentRunner:
        """Access AI agents.

        Returns:
            AgentRunner for fluent agent access

        Raises:
            NotInitializedError: If runtime not initialized

        Example:
            response = await runtime.agents.codebase_qna_agent.query(ctx)
            async for chunk in runtime.agents.debugging_agent.stream(ctx):
                print(chunk.response, end="")
        """
        if not self._initialized:
            raise NotInitializedError(
                "Runtime not initialized - call initialize() first"
            )

        if self._agents is None:
            from potpie.agents.runner import AgentRunner
            from app.modules.intelligence.provider.provider_service import (
                ProviderService,
            )
            from app.modules.intelligence.tools.tool_service import ToolService
            from app.modules.intelligence.prompts.prompt_service import PromptService

            self._agent_session = self._db_manager.get_session()

            provider_config = {
                "provider": self._config.llm_provider,
                "api_key": self._config.llm_api_key,
                "chat_model": self._config.llm_chat_model,
                "inference_model": self._config.llm_inference_model,
                "base_url": self._config.llm_base_url,
            }

            provider_service = ProviderService.create_from_config(
                self._agent_session,
                self._config.default_user_id,
                **provider_config,
            )

            tool_service = ToolService(
                self._agent_session, self._config.default_user_id
            )
            prompt_service = PromptService(self._agent_session)

            self._agents = AgentRunner(
                db_session=self._agent_session,
                user_id=self._config.default_user_id,
                provider_service=provider_service,
                tool_service=tool_service,
                prompt_service=prompt_service,
                provider_config=provider_config,
            )

        return self._agents

    async def initialize(self) -> None:
        """Initialize all runtime components.

        Must be called before using the runtime unless using async context manager.
        """
        if self._initialized:
            return

        logger.info("Initializing PotpieRuntime...")

        try:
            # Initialize database manager
            self._db_manager = DatabaseManager(self._config)
            await self._db_manager.initialize()

            # Initialize Neo4j manager
            self._neo4j_manager = Neo4jManager(self._config)
            await self._neo4j_manager.initialize()

            # Initialize Redis manager (optional)
            self._redis_manager = RedisManager(self._config)
            await self._redis_manager.initialize()

            self._initialized = True
            logger.info("PotpieRuntime initialized successfully")

        except Exception as e:
            # Clean up on failure
            await self._cleanup()
            raise PotpieError(f"Failed to initialize runtime: {e}") from e

    async def close(self) -> None:
        """Close all runtime components and release resources."""
        logger.info("Closing PotpieRuntime...")
        await self._cleanup()
        logger.info("PotpieRuntime closed")

    async def _cleanup(self) -> None:
        """Internal cleanup of all managers."""
        # Close agent session if it exists
        if self._agent_session:
            try:
                self._agent_session.close()
            except Exception as e:
                logger.warning(f"Error closing agent session: {e}")
            self._agent_session = None

        # Clear agent runner reference
        self._agents = None

        if self._redis_manager:
            try:
                await self._redis_manager.close()
            except Exception as e:
                logger.warning(f"Error closing Redis manager: {e}")
            self._redis_manager = None

        if self._neo4j_manager:
            try:
                await self._neo4j_manager.close()
            except Exception as e:
                logger.warning(f"Error closing Neo4j manager: {e}")
            self._neo4j_manager = None

        if self._db_manager:
            try:
                await self._db_manager.close()
            except Exception as e:
                logger.warning(f"Error closing database manager: {e}")
            self._db_manager = None

        self._initialized = False

    async def verify_connections(self) -> dict:
        """Verify all connections are working.

        Returns:
            Dictionary with connection status for each component

        Raises:
            NotInitializedError: If runtime not initialized
        """
        if not self._initialized:
            raise NotInitializedError("Runtime not initialized")

        results = {}

        try:
            await self._db_manager.verify_connection()
            results["postgres"] = {"status": "connected"}
        except Exception as e:
            results["postgres"] = {"status": "error", "error": str(e)}

        try:
            await self._neo4j_manager.verify_connection()
            results["neo4j"] = {"status": "connected"}
        except Exception as e:
            results["neo4j"] = {"status": "error", "error": str(e)}

        try:
            if await self._redis_manager.verify_connection():
                results["redis"] = {"status": "connected"}
            else:
                results["redis"] = {"status": "not_configured"}
        except Exception as e:
            results["redis"] = {"status": "error", "error": str(e)}

        return results

    async def __aenter__(self) -> PotpieRuntime:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        return f"<PotpieRuntime status={status}>"


class _RuntimeBuilder(RuntimeConfigBuilder):
    """Extended builder that can directly create PotpieRuntime."""

    def build_runtime(self) -> PotpieRuntime:
        """Build and return a PotpieRuntime instance.

        Returns:
            PotpieRuntime configured with this builder's settings
        """
        config = self.build()
        return PotpieRuntime(config)
