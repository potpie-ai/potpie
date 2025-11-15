"""
Redis-based distributed semaphore for global rate limiting across Celery workers.

This module provides an atomic, race-condition-free semaphore implementation using
Redis Lua scripts to coordinate concurrent access to rate-limited resources (e.g., OpenAI API).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class RedisDistributedSemaphore:
    """
    Distributed semaphore using Redis with atomic Lua scripts.

    This semaphore prevents race conditions by using atomic Lua scripts
    that execute on the Redis server, ensuring check-then-act operations
    are atomic.

    Key features:
    - Atomic acquire/release operations (no race conditions)
    - Automatic TTL management to prevent leaked permits
    - Configurable timeout for acquire attempts
    - Async context manager support for clean resource management

    Example:
        ```python
        semaphore = RedisDistributedSemaphore(
            redis_client=redis_client,
            key="openai:inference:semaphore",
            max_concurrent=50,
            ttl=300  # 5 minutes
        )

        async with semaphore.acquire(timeout=60):
            # Make OpenAI API call
            response = await openai_client.create(...)
        ```
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        key: str,
        max_concurrent: int,
        ttl: int = 300,  # 5 minutes default TTL
    ):
        """
        Initialize the distributed semaphore.

        Args:
            redis_client: Redis client instance
            key: Redis key for the semaphore counter
            max_concurrent: Maximum number of concurrent permits allowed
            ttl: Time-to-live in seconds for the semaphore key (prevents leaks)
        """
        self.redis_client = redis_client
        self.key = key
        self.max_concurrent = max_concurrent
        self.ttl = ttl

        # Register atomic Lua scripts
        # ACQUIRE script: atomically check and increment if below max
        self.acquire_script = self.redis_client.register_script("""
            local key = KEYS[1]
            local max_value = tonumber(ARGV[1])
            local ttl = tonumber(ARGV[2])

            local current = redis.call('GET', key)
            current = tonumber(current) or 0

            if current < max_value then
                local new_value = redis.call('INCR', key)
                -- Set TTL only on first acquire to prevent expiration during active use
                if new_value == 1 then
                    redis.call('EXPIRE', key, ttl)
                end
                return new_value
            else
                return -1  -- Semaphore full
            end
        """)

        # RELEASE script: atomically decrement and clean up if zero
        self.release_script = self.redis_client.register_script("""
            local key = KEYS[1]

            local current = redis.call('GET', key)
            current = tonumber(current) or 0

            if current > 0 then
                local new_value = redis.call('DECR', key)
                -- Delete key if no more permits held (cleanup)
                if new_value == 0 then
                    redis.call('DEL', key)
                end
                return new_value
            else
                return 0  -- Already at zero
            end
        """)

        logger.info(
            f"Initialized RedisDistributedSemaphore: key={key}, "
            f"max_concurrent={max_concurrent}, ttl={ttl}s"
        )

    def try_acquire(self) -> bool:
        """
        Attempt to acquire a permit from the semaphore (non-blocking).

        Returns:
            True if permit acquired, False if semaphore is full
        """
        try:
            result = self.acquire_script(
                keys=[self.key],
                args=[self.max_concurrent, self.ttl]
            )

            if result == -1:
                logger.debug(f"Semaphore full: {self.key}")
                return False
            else:
                logger.debug(f"Acquired permit: {self.key}, current={result}")
                return True

        except Exception as e:
            logger.error(f"Error acquiring semaphore {self.key}: {e}")
            # On error, fail open (return True) to avoid blocking workers indefinitely
            return True

    def release(self) -> None:
        """
        Release a permit back to the semaphore.

        This should always be called after acquiring, typically in a finally block
        or via the async context manager.
        """
        try:
            result = self.release_script(keys=[self.key])
            logger.debug(f"Released permit: {self.key}, current={result}")

        except Exception as e:
            logger.error(f"Error releasing semaphore {self.key}: {e}")

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None, poll_interval: float = 0.1):
        """
        Async context manager for acquiring and releasing the semaphore.

        Args:
            timeout: Maximum time to wait for a permit (seconds). None = wait forever
            poll_interval: Time to wait between acquire attempts (seconds)

        Raises:
            TimeoutError: If timeout is reached without acquiring permit

        Example:
            ```python
            async with semaphore.acquire(timeout=60):
                # Critical section - OpenAI API call
                pass
            ```
        """
        start_time = time.time()
        acquired = False

        try:
            # Poll until we acquire or timeout
            while True:
                if self.try_acquire():
                    acquired = True
                    break

                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise TimeoutError(
                            f"Failed to acquire semaphore {self.key} within {timeout}s"
                        )

                # Wait before next attempt (async sleep to avoid blocking)
                import asyncio
                await asyncio.sleep(poll_interval)

            # Yield control to caller (critical section)
            yield

        finally:
            # Always release if we acquired
            if acquired:
                self.release()

    def get_current_count(self) -> int:
        """
        Get the current number of permits in use.

        Returns:
            Number of permits currently held (0 if key doesn't exist)
        """
        try:
            value = self.redis_client.get(self.key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Error getting semaphore count {self.key}: {e}")
            return 0

    def reset(self) -> None:
        """
        Reset the semaphore to zero (emergency cleanup).

        WARNING: Only use this for testing or emergency recovery.
        This will break active permit holders.
        """
        try:
            self.redis_client.delete(self.key)
            logger.warning(f"Reset semaphore: {self.key}")
        except Exception as e:
            logger.error(f"Error resetting semaphore {self.key}: {e}")


# Global connection pool (singleton)
_redis_pool: Optional[redis.ConnectionPool] = None


def get_redis_pool() -> redis.ConnectionPool:
    """
    Get or create a global Redis connection pool.

    Uses a singleton pattern to ensure only one connection pool exists
    across all workers, preventing connection exhaustion.

    Returns:
        Redis connection pool instance
    """
    global _redis_pool

    if _redis_pool is None:
        # Get Redis connection info from environment
        redishost = os.getenv("REDISHOST", "localhost")
        redisport = int(os.getenv("REDISPORT", 6379))
        redisuser = os.getenv("REDISUSER", "")
        redispassword = os.getenv("REDISPASSWORD", "")

        # Create connection pool with limits
        _redis_pool = redis.ConnectionPool(
            host=redishost,
            port=redisport,
            username=redisuser if redisuser else None,
            password=redispassword if redispassword else None,
            db=0,
            decode_responses=False,  # Keep bytes for Lua scripts
            max_connections=100,  # Limit max connections
        )

        logger.info(
            f"Created Redis connection pool: host={redishost}, "
            f"port={redisport}, max_connections=100"
        )

    return _redis_pool


def get_redis_semaphore(
    key_suffix: str = "inference",
    max_concurrent: Optional[int] = None,
    ttl: int = 300,
) -> RedisDistributedSemaphore:
    """
    Factory function to create a RedisDistributedSemaphore with default configuration.

    Uses a shared connection pool to avoid connection leaks.

    Args:
        key_suffix: Suffix for the Redis key (full key: "momentum:semaphore:{suffix}")
        max_concurrent: Max concurrent permits (defaults to MAX_GLOBAL_LLM_REQUESTS env var)
        ttl: Time-to-live in seconds

    Returns:
        Configured RedisDistributedSemaphore instance
    """
    # Get shared connection pool
    pool = get_redis_pool()

    # Create Redis client from pool
    redis_client = redis.Redis(connection_pool=pool)

    # Get max concurrent from environment or use default
    if max_concurrent is None:
        max_concurrent = int(os.getenv("MAX_GLOBAL_LLM_REQUESTS", "50"))

    # Create semaphore with namespaced key
    key = f"momentum:semaphore:{key_suffix}"

    return RedisDistributedSemaphore(
        redis_client=redis_client,
        key=key,
        max_concurrent=max_concurrent,
        ttl=ttl,
    )
