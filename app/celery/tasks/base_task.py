import logging
from celery import Task
from contextlib import asynccontextmanager
from app.core.database import SessionLocal, AsyncSessionLocal

logger = logging.getLogger(__name__)


class BaseTask(Task):
    _db = None
    _async_db = None

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
        if self._async_db is None:
            self._async_db = AsyncSessionLocal()

        try:
            # Yield the session to the `async with` block in the task
            yield self._async_db
        finally:
            # This code runs after the `async with` block exits
            if self._async_db:
                await self._async_db.close()
                self._async_db = None

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
