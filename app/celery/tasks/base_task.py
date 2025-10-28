import logging
import asyncio
from celery import Task
from contextlib import asynccontextmanager
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


class BaseTask(Task):
    _db = None
    _loop = None

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
        Returns a long-lived event loop for this worker process. Creates one if needed.
        """
        # Reuse a single loop per worker process to avoid cross-loop issues
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def run_async(self, coro):
        """
        Run the given coroutine on the worker's long-lived event loop.
        """
        loop = self._get_event_loop()
        return loop.run_until_complete(coro)

    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task execution."""
        logger.info(f"Task {task_id} completed successfully")
        if self._db:
            self._db.close()
            self._db = None

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        if self._db:
            self._db.close()
            self._db = None

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning(f"Task {task_id} retrying: {exc}")
