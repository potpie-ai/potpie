"""Database manager for PostgreSQL connections."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from potpie.exceptions import DatabaseError, NotInitializedError

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections for the runtime.

    Provides both sync and async session factories with configurable pooling.
    """

    def __init__(self, config: RuntimeConfig):
        """Initialize the database manager.

        Args:
            config: Runtime configuration with database settings
        """
        self._config = config
        self._engine: Optional[object] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[sessionmaker] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the database manager has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize database connections and session factories.

        Creates both sync and async engines with configured pool settings.
        """
        if self._initialized:
            return

        try:
            # Import all SQLAlchemy models to register them and resolve forward references
            # This must be done before creating any sessions
            import potpie.core.models  # noqa

            # Create sync engine
            self._engine = create_engine(
                self._config.postgres_url,
                pool_size=self._config.db_pool_size,
                max_overflow=self._config.db_max_overflow,
                pool_timeout=self._config.db_pool_timeout,
                pool_recycle=self._config.db_pool_recycle,
                pool_pre_ping=True,
                echo=False,
            )

            # Create sync session factory
            self._session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine,
            )

            # Create async engine
            async_url = self._config.postgres_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            self._async_engine = create_async_engine(
                async_url,
                pool_size=self._config.db_pool_size,
                max_overflow=self._config.db_max_overflow,
                pool_timeout=self._config.db_pool_timeout,
                pool_recycle=self._config.db_pool_recycle,
                pool_pre_ping=False,  # Disabled for async to avoid event loop issues
                echo=False,
            )

            # Create async session factory
            self._async_session_factory = sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )

            self._initialized = True
            logger.info("Database manager initialized successfully")

        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    async def close(self) -> None:
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None

        if self._engine:
            self._engine.dispose()
            self._engine = None

        self._session_factory = None
        self._async_session_factory = None
        self._initialized = False
        logger.info("Database manager closed")

    async def verify_connection(self) -> bool:
        """Verify database connectivity.

        Returns:
            True if connection is successful

        Raises:
            DatabaseError: If connection fails
        """
        if not self._initialized:
            raise NotInitializedError("Database manager not initialized")

        try:
            async with self.async_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            raise DatabaseError(f"Database connection verification failed: {e}") from e

    def get_session(self) -> Session:
        """Get a new sync database session.

        Returns:
            SQLAlchemy Session

        Raises:
            NotInitializedError: If manager not initialized
        """
        if not self._initialized or self._session_factory is None:
            raise NotInitializedError("Database manager not initialized")
        return self._session_factory()

    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session as context manager.

        Yields:
            AsyncSession that auto-closes on exit

        Raises:
            NotInitializedError: If manager not initialized
        """
        if not self._initialized or self._async_session_factory is None:
            raise NotInitializedError("Database manager not initialized")

        session = self._async_session_factory()
        try:
            yield session
        finally:
            await session.close()

    def create_isolated_session(self) -> tuple[AsyncSession, AsyncEngine]:
        """Create an isolated async session with its own non-pooled connection.

        Useful for long-running operations that shouldn't hold pool connections.

        Returns:
            Tuple of (session, engine) - caller must close both

        Raises:
            NotInitializedError: If manager not initialized
        """
        if not self._initialized:
            raise NotInitializedError("Database manager not initialized")

        async_url = self._config.postgres_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )

        engine = create_async_engine(
            async_url,
            poolclass=NullPool,
            echo=False,
        )

        factory = sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

        return factory(), engine
