"""Redis manager for optional caching and streaming."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from potpie.exceptions import NotInitializedError, RedisError

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig

logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis connections for caching and streaming.

    This is optional - the runtime can work without Redis.
    """

    def __init__(self, config: RuntimeConfig):
        """Initialize the Redis manager.

        Args:
            config: Runtime configuration with Redis settings
        """
        self._config = config
        self._client: Optional[Any] = None  # redis.Redis
        self._async_client: Optional[Any] = None  # redis.asyncio.Redis
        self._initialized = False
        self._available = False

    @property
    def is_initialized(self) -> bool:
        """Check if the Redis manager has been initialized."""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Check if Redis is available and connected."""
        return self._available

    async def initialize(self) -> None:
        """Initialize Redis connections.

        If Redis URL is not configured, marks as unavailable but doesn't fail.
        """
        if self._initialized:
            return

        if not self._config.redis_url:
            logger.info("Redis URL not configured - Redis features disabled")
            self._initialized = True
            self._available = False
            return

        try:
            import redis.asyncio as aioredis

            self._async_client = aioredis.from_url(
                self._config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

            # Test connection
            await self._async_client.ping()

            self._initialized = True
            self._available = True
            logger.info("Redis manager initialized successfully")

        except ImportError:
            logger.warning("redis package not installed - Redis features disabled")
            self._initialized = True
            self._available = False

        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - Redis features disabled")
            self._initialized = True
            self._available = False

    async def close(self) -> None:
        """Close Redis connections."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

        self._initialized = False
        self._available = False
        logger.info("Redis manager closed")

    async def verify_connection(self) -> bool:
        """Verify Redis connectivity.

        Returns:
            True if connected, False if not available

        Raises:
            RedisError: If Redis was configured but connection fails
        """
        if not self._initialized:
            raise NotInitializedError("Redis manager not initialized")

        if not self._available:
            return False

        try:
            await self._async_client.ping()
            return True
        except Exception as e:
            raise RedisError(f"Redis connection verification failed: {e}") from e

    async def get(self, key: str) -> Optional[str]:
        """Get a value from Redis.

        Args:
            key: Redis key

        Returns:
            Value if found, None otherwise

        Raises:
            NotInitializedError: If not initialized
            RedisError: If Redis unavailable but was configured
        """
        if not self._initialized:
            raise NotInitializedError("Redis manager not initialized")

        if not self._available:
            return None

        try:
            return await self._async_client.get(key)
        except Exception as e:
            raise RedisError(f"Redis get failed: {e}") from e

    async def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
    ) -> None:
        """Set a value in Redis.

        Args:
            key: Redis key
            value: Value to set
            ex: Optional expiration in seconds

        Raises:
            NotInitializedError: If not initialized
            RedisError: If operation fails
        """
        if not self._initialized:
            raise NotInitializedError("Redis manager not initialized")

        if not self._available:
            logger.debug("Redis not available - set operation skipped")
            return

        try:
            await self._async_client.set(key, value, ex=ex)
        except Exception as e:
            raise RedisError(f"Redis set failed: {e}") from e

    async def delete(self, key: str) -> None:
        """Delete a key from Redis.

        Args:
            key: Redis key

        Raises:
            NotInitializedError: If not initialized
        """
        if not self._initialized:
            raise NotInitializedError("Redis manager not initialized")

        if not self._available:
            return

        try:
            await self._async_client.delete(key)
        except Exception as e:
            raise RedisError(f"Redis delete failed: {e}") from e

    def get_url(self) -> Optional[str]:
        """Get the Redis URL if configured.

        Returns:
            Redis URL or None if not configured
        """
        return self._config.redis_url
