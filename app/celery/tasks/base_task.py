import logging
import asyncio
from celery import Task
from contextlib import asynccontextmanager
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


class BaseTask(Task):
    # Use instance-level attributes instead of class-level to avoid sharing across tasks
    # Each task instance will have its own db session and event loop
    abstract = True  # Mark as abstract so Celery doesn't try to register it directly

    def __init__(self):
        super().__init__()
        self._db = None
        self._loop = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    @asynccontextmanager
    async def async_db(self):
        """
        Provides an async session with a fresh connection for Celery tasks.

        This creates a non-pooled connection to avoid asyncpg Future binding issues
        when tasks share the same event loop but have different coroutine contexts.

        Usage:
            async with self.async_db() as session:
                result = await session.execute(query)
                await session.commit()
        """
        from app.core.database import create_celery_async_session

        try:
            task_id = self.request.id if self.request else "test"
        except (AttributeError, TypeError):
            task_id = "test"

        logger.debug(f"[Task {task_id}] Creating fresh async DB connection")
        async_session, engine = create_celery_async_session()

        try:
            yield async_session
            logger.debug(
                f"[Task {task_id}] Async DB session operation completed successfully"
            )
        except Exception as e:
            logger.error(
                f"[Task {task_id}] Error during async DB operation: {e}", exc_info=True
            )
            raise
        finally:
            try:
                await async_session.close()
                if engine is not None:
                    await engine.dispose()
                logger.debug(
                    f"[Task {task_id}] Async DB connection closed and engine disposed"
                )
            except Exception as cleanup_error:
                logger.error(
                    f"[Task {task_id}] Error during connection cleanup: {cleanup_error}",
                    exc_info=True,
                )

    def _get_event_loop(self):
        """
        Returns a fresh event loop for each task execution.
        This ensures tasks don't block each other when running concurrently.
        """
        # Create a new event loop for each task to enable true concurrency
        # Each task gets its own loop, so run_until_complete() doesn't block other tasks
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            # Don't set as the global loop - keep it task-local
        return self._loop

    def run_async(self, coro):
        """
        Run the given coroutine on the task's own event loop.
        """
        loop = self._get_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up the loop after use to prevent resource leaks
            pass  # Keep loop alive for potential reuse within same task

    def on_success(self, retval, task_id, args, kwargs):
        try:
            status = "cancelled" if retval is False else "completed successfully"
            logger.info(f"Task {task_id} {status}")
        finally:
            self._cleanup_resources()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        self._cleanup_resources()

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning(f"Task {task_id} retrying: {exc}")

    def _cleanup_resources(self):
        """Clean up task-local resources (db session and event loop)."""
        if self._db:
            self._db.close()
            self._db = None
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")
            self._loop = None
