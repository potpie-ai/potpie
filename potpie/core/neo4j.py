"""Neo4j manager for knowledge graph connections."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from potpie.exceptions import Neo4jError, NotInitializedError

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manages Neo4j database connections for the knowledge graph.

    Provides driver management and session context for Neo4j operations.
    """

    def __init__(self, config: RuntimeConfig):
        """Initialize the Neo4j manager.

        Args:
            config: Runtime configuration with Neo4j settings
        """
        self._config = config
        self._driver: Optional[AsyncDriver] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the Neo4j manager has been initialized."""
        return self._initialized

    @property
    def driver(self) -> AsyncDriver:
        """Get the Neo4j driver.

        Returns:
            Neo4j AsyncDriver instance

        Raises:
            NotInitializedError: If manager not initialized
        """
        if not self._initialized or self._driver is None:
            raise NotInitializedError("Neo4j manager not initialized")
        return self._driver

    async def initialize(self) -> None:
        """Initialize Neo4j driver connection.

        Creates the driver with configured credentials.
        """
        if self._initialized:
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self._config.neo4j_uri,
                auth=(self._config.neo4j_username, self._config.neo4j_password),
            )
            self._initialized = True
            logger.info("Neo4j manager initialized successfully")

        except Exception as e:
            raise Neo4jError(f"Failed to initialize Neo4j: {e}") from e

    async def close(self) -> None:
        """Close Neo4j driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

        self._initialized = False
        logger.info("Neo4j manager closed")

    async def verify_connection(self) -> bool:
        """Verify Neo4j connectivity.

        Returns:
            True if connection is successful

        Raises:
            Neo4jError: If connection fails
        """
        if not self._initialized or self._driver is None:
            raise NotInitializedError("Neo4j manager not initialized")

        try:
            async with self._driver.session() as session:
                result = await session.run("RETURN 1")
                await result.consume()
            return True
        except Exception as e:
            raise Neo4jError(f"Neo4j connection verification failed: {e}") from e

    @asynccontextmanager
    async def session(
        self, database: Optional[str] = None
    ) -> AsyncGenerator[AsyncSession, None]:
        """Get a Neo4j session as async context manager.

        Args:
            database: Optional database name (uses default if None)

        Yields:
            Neo4j AsyncSession that auto-closes on exit

        Raises:
            NotInitializedError: If manager not initialized
        """
        if not self._initialized or self._driver is None:
            raise NotInitializedError("Neo4j manager not initialized")

        session = (
            self._driver.session(database=database)
            if database
            else self._driver.session()
        )
        try:
            yield session
        finally:
            await session.close()

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Optional query parameters
            database: Optional database name

        Returns:
            List of result records as dictionaries

        Raises:
            NotInitializedError: If manager not initialized
            Neo4jError: If query execution fails
        """
        if not self._initialized or self._driver is None:
            raise NotInitializedError("Neo4j manager not initialized")

        try:
            async with self.session(database=database) as session:
                result = await session.run(query, parameters or {})
                records = []
                async for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            raise Neo4jError(f"Query execution failed: {e}") from e

    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> None:
        """Execute a Cypher write query within a transaction.

        Args:
            query: Cypher query string
            parameters: Optional query parameters
            database: Optional database name

        Raises:
            NotInitializedError: If manager not initialized
            Neo4jError: If query execution fails
        """
        if not self._initialized or self._driver is None:
            raise NotInitializedError("Neo4j manager not initialized")

        async def _write_tx(tx):
            await tx.run(query, parameters or {})

        try:
            async with self.session(database=database) as session:
                await session.execute_write(_write_tx)
        except Exception as e:
            raise Neo4jError(f"Write query execution failed: {e}") from e

    def get_neo4j_config(self) -> Dict[str, str]:
        """Get Neo4j configuration dictionary.

        Returns config in the format expected by existing Potpie services.

        Returns:
            Dictionary with uri, username, password keys
        """
        return {
            "uri": self._config.neo4j_uri,
            "username": self._config.neo4j_username,
            "password": self._config.neo4j_password,
        }
