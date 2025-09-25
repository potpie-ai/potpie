import logging
from celery import Task
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


class BaseTask(Task):
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

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
