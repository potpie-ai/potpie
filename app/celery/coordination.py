import logging
from typing import Optional, Tuple
from redis import Redis

logger = logging.getLogger(__name__)


class ParsingCoordinator:
    """Coordinates distributed parsing using Redis atomic operations"""

    @staticmethod
    def _get_namespace(project_id: str, commit_id: Optional[str]) -> str:
        """
        Get Redis key namespace for a project/commit.

        Args:
            project_id: Project identifier
            commit_id: Commit identifier (None for projects without commits)

        Returns:
            Namespace string safe for Redis keys
        """
        commit_part = commit_id if commit_id else "none"
        return f"parsing:{project_id}:{commit_part}"

    @staticmethod
    def mark_work_unit_completed(
        redis_client: Redis,
        project_id: str,
        commit_id: Optional[str],
        work_unit_id: str
    ) -> bool:
        """
        Idempotently mark a work unit as completed using Redis SET.

        This prevents double-counting when tasks are retried due to
        worker crashes or acknowledgment delays.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            commit_id: Commit identifier (None for projects without commits)
            work_unit_id: Unique work unit identifier (UUID)

        Returns:
            True if this is the first completion, False if already completed
        """
        namespace = ParsingCoordinator._get_namespace(project_id, commit_id)
        set_key = f"{namespace}:completed_units"

        # SADD returns 1 if element was added (new), 0 if already exists
        is_new = redis_client.sadd(set_key, work_unit_id)
        redis_client.expire(set_key, 86400)  # 24 hour TTL

        if is_new:
            logger.info(
                f"Marked work unit {work_unit_id} as completed "
                f"(project={project_id}, commit={commit_id or 'none'})"
            )
        else:
            logger.warning(
                f"Work unit {work_unit_id} already marked completed - duplicate execution detected "
                f"(project={project_id}, commit={commit_id or 'none'})"
            )

        return bool(is_new)

    @staticmethod
    def increment_completed(
        redis_client: Redis,
        project_id: str,
        commit_id: Optional[str],
        total_work_units: int,
        work_unit_id: Optional[str] = None
    ) -> Tuple[int, bool]:
        """
        Atomically increment completed counter and check if last worker.

        If work_unit_id is provided, uses idempotent SET-based tracking
        to prevent double-counting on retries.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            commit_id: Commit identifier (None for projects without commits)
            total_work_units: Total number of work units expected
            work_unit_id: Optional work unit ID for idempotent tracking

        Returns:
            tuple: (completed_count, is_last_worker)
        """
        namespace = ParsingCoordinator._get_namespace(project_id, commit_id)

        # If work_unit_id provided, check if already completed
        if work_unit_id:
            is_new = ParsingCoordinator.mark_work_unit_completed(
                redis_client, project_id, commit_id, work_unit_id
            )
            if not is_new:
                # Already counted, return current count without incrementing
                key = f"{namespace}:completed"
                current = redis_client.get(key)
                completed = int(current) if current else 0
                is_last = completed >= total_work_units
                return completed, is_last

        # Increment counter
        key = f"{namespace}:completed"
        completed = redis_client.incr(key)
        redis_client.expire(key, 86400)  # 24 hour TTL

        is_last = completed >= total_work_units

        if is_last:
            logger.info(
                f"Last worker detected for project {project_id}, commit {commit_id or 'none'}: "
                f"{completed}/{total_work_units}"
            )

        return completed, is_last

    @staticmethod
    def reset_counter(redis_client: Redis, project_id: str, commit_id: Optional[str] = None):
        """
        Reset completion counter and completed units set (for retries/cleanup).

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            commit_id: Commit identifier (None for projects without commits)
        """
        namespace = ParsingCoordinator._get_namespace(project_id, commit_id)
        counter_key = f"{namespace}:completed"
        set_key = f"{namespace}:completed_units"

        deleted_count = redis_client.delete(counter_key)
        deleted_set = redis_client.delete(set_key)

        if deleted_count or deleted_set:
            logger.info(
                f"Reset completion tracking for project {project_id}, commit {commit_id or 'none'} "
                f"(counter={'yes' if deleted_count else 'no'}, set={'yes' if deleted_set else 'no'})"
            )

    @staticmethod
    def get_count(redis_client: Redis, project_id: str, commit_id: Optional[str] = None) -> int:
        """
        Get current completion count without incrementing.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            commit_id: Commit identifier (None for projects without commits)

        Returns:
            Current count (0 if key doesn't exist)
        """
        namespace = ParsingCoordinator._get_namespace(project_id, commit_id)
        key = f"{namespace}:completed"
        count = redis_client.get(key)
        return int(count) if count else 0


class InferenceCoordinator:
    """Coordinates distributed inference using Redis atomic operations (same pattern as ParsingCoordinator)"""

    @staticmethod
    def _get_namespace(project_id: str, session_id: str) -> str:
        """
        Get Redis key namespace for a project/session.

        Args:
            project_id: Project identifier
            session_id: Inference session identifier

        Returns:
            Namespace string safe for Redis keys
        """
        return f"inference:{project_id}:{session_id}"

    @staticmethod
    def mark_work_unit_completed(
        redis_client: Redis,
        project_id: str,
        session_id: str,
        work_unit_id: str
    ) -> bool:
        """
        Idempotently mark a work unit as completed using Redis SET.

        This prevents double-counting when tasks are retried due to
        worker crashes or acknowledgment delays.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            session_id: Inference session identifier
            work_unit_id: Unique work unit identifier (UUID)

        Returns:
            True if this is the first completion, False if already completed
        """
        namespace = InferenceCoordinator._get_namespace(project_id, session_id)
        set_key = f"{namespace}:completed_units"

        # SADD returns 1 if element was added (new), 0 if already exists
        is_new = redis_client.sadd(set_key, work_unit_id)
        redis_client.expire(set_key, 86400)  # 24 hour TTL

        if is_new:
            logger.info(
                f"[Inference] Marked work unit {work_unit_id} as completed "
                f"(project={project_id}, session={session_id})"
            )
        else:
            logger.warning(
                f"[Inference] Work unit {work_unit_id} already marked completed - duplicate execution "
                f"(project={project_id}, session={session_id})"
            )

        return bool(is_new)

    @staticmethod
    def increment_completed(
        redis_client: Redis,
        project_id: str,
        session_id: str,
        total_work_units: int,
        work_unit_id: Optional[str] = None
    ) -> Tuple[int, bool]:
        """
        Atomically increment completed counter and check if last worker.

        If work_unit_id is provided, uses idempotent SET-based tracking
        to prevent double-counting on retries.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            session_id: Inference session identifier
            total_work_units: Total number of work units expected
            work_unit_id: Optional work unit ID for idempotent tracking

        Returns:
            tuple: (completed_count, is_last_worker)
        """
        namespace = InferenceCoordinator._get_namespace(project_id, session_id)

        # If work_unit_id provided, check if already completed
        if work_unit_id:
            is_new = InferenceCoordinator.mark_work_unit_completed(
                redis_client, project_id, session_id, work_unit_id
            )
            if not is_new:
                # Already counted, return current count without incrementing
                key = f"{namespace}:completed"
                current = redis_client.get(key)
                completed = int(current) if current else 0
                is_last = completed >= total_work_units
                return completed, is_last

        # Increment counter
        key = f"{namespace}:completed"
        completed = redis_client.incr(key)
        redis_client.expire(key, 86400)  # 24 hour TTL

        is_last = completed >= total_work_units

        if is_last:
            logger.info(
                f"[Inference] Last worker detected for project {project_id}, session {session_id}: "
                f"{completed}/{total_work_units}"
            )

        return completed, is_last

    @staticmethod
    def reset_counter(redis_client: Redis, project_id: str, session_id: str):
        """
        Reset completion counter and completed units set (for retries/cleanup).

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            session_id: Inference session identifier
        """
        namespace = InferenceCoordinator._get_namespace(project_id, session_id)
        counter_key = f"{namespace}:completed"
        set_key = f"{namespace}:completed_units"

        deleted_count = redis_client.delete(counter_key)
        deleted_set = redis_client.delete(set_key)

        if deleted_count or deleted_set:
            logger.info(
                f"[Inference] Reset completion tracking for project {project_id}, session {session_id} "
                f"(counter={'yes' if deleted_count else 'no'}, set={'yes' if deleted_set else 'no'})"
            )

    @staticmethod
    def get_count(redis_client: Redis, project_id: str, session_id: str) -> int:
        """
        Get current completion count without incrementing.

        Args:
            redis_client: Redis client instance
            project_id: Project identifier
            session_id: Inference session identifier

        Returns:
            Current count (0 if key doesn't exist)
        """
        namespace = InferenceCoordinator._get_namespace(project_id, session_id)
        key = f"{namespace}:completed"
        count = redis_client.get(key)
        return int(count) if count else 0
