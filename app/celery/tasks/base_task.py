import logging
import asyncio
from celery import Task
from contextlib import asynccontextmanager
from app.core.database import SessionLocal, AsyncSessionLocal

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
        Provides an async session that is automatically created and closed.
        """
        async_session = AsyncSessionLocal()
        try:
            yield async_session
        finally:
            await async_session.close()

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
