import logging
from typing import Optional, Tuple
from redis import Redis

logger = logging.getLogger(__name__)


class ParsingCoordinator:
    """Coordinates distributed parsing using Redis atomic operations"""

    @staticmethod
    def increment_completed(
        redis_client: Redis,
        project_id: str,
        total_work_units: int
    ) -> Tuple[int, bool]:
        """
        Atomically increment completed counter and check if last worker.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            total_work_units: Total number of work units expected

        Returns:
            tuple: (completed_count, is_last_worker)
        """
        key = f"parsing:{project_id}:completed"
        completed = redis_client.incr(key)
        redis_client.expire(key, 86400)  # 24 hour TTL

        is_last = completed >= total_work_units

        if is_last:
            logger.info(
                f"Last worker detected for project {project_id}: "
                f"{completed}/{total_work_units}"
            )

        return completed, is_last

    @staticmethod
    def reset_counter(redis_client: Redis, project_id: str):
        """
        Reset completion counter (for retries/cleanup).

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
        """
        key = f"parsing:{project_id}:completed"
        deleted = redis_client.delete(key)
        if deleted:
            logger.info(f"Reset completion counter for project {project_id}")

    @staticmethod
    def get_count(redis_client: Redis, project_id: str) -> int:
        """
        Get current completion count without incrementing.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier

        Returns:
            Current count (0 if key doesn't exist)
        """
        key = f"parsing:{project_id}:completed"
        count = redis_client.get(key)
        return int(count) if count else 0
