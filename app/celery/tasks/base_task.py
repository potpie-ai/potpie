import asyncio
from celery import Task
from contextlib import asynccontextmanager
from app.core.database import SessionLocal
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseTask(Task):
    """
    Base for Celery tasks that may use async (run_async) and sync DB.

    Async/sync rule (avoids deadlocks and hangs in forked workers):
    - Do ALL sync DB (SessionLocal, self.db) and any blocking I/O BEFORE calling
      run_async(). Resolve user, load config, etc. in the sync task body.
    - Inside the coroutine passed to run_async(), use ONLY async_db() for DB;
      never call SessionLocal() or use self.db. Using sync DB inside the
      coroutine blocks the event loop and can hang (e.g. pool/connection state
      in forked workers).
    """

    _db = None

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
        when each task runs in a fresh event loop via asyncio.run().

        Use this only inside the coroutine passed to run_async(). Do not use sync
        DB (SessionLocal or self.db) inside that coroutine — do sync work before
        run_async() and pass results in.

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

        logger.debug("Creating fresh async DB connection", task_id=task_id)
        async_session, engine = create_celery_async_session()

        try:
            yield async_session
            logger.debug(
                "Async DB session operation completed successfully", task_id=task_id
            )
        except Exception:
            logger.exception("Error during async DB operation", task_id=task_id)
            raise
        finally:
            try:
                await async_session.close()
                if engine is not None:
                    await engine.dispose()
                logger.debug(
                    "Async DB connection closed and engine disposed", task_id=task_id
                )
            except Exception:
                logger.exception("Error during connection cleanup", task_id=task_id)

    def run_async(self, coro):
        """
        Run the given coroutine in a fresh event loop per invocation.
        asyncio.run() calls shutdown_asyncgens() before closing the loop,
        ensuring all pydantic_ai async generators are properly finalized in
        the correct context — preventing deferred aclose() callbacks from
        one agent run leaking into the next and causing OTel context errors.

        Important: the coroutine must not perform sync DB (SessionLocal or
        self.db) or other blocking I/O; do that before calling run_async()
        and pass results into the coroutine (see class docstring).
        """
        return asyncio.run(coro)

    def on_success(self, retval, task_id, args, kwargs):
        try:
            status = "cancelled" if retval is False else "completed successfully"
            logger.info("Task completed", task_id=task_id, status=status)
        finally:
            if self._db:
                self._db.close()  # Returns to pool
                self._db = None

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        # exc is already an exception object from on_failure
        logger.error(
            "Task failed",
            task_id=task_id,
            error=str(exc),
            exc_info=einfo.exc_info if einfo else None,
        )
        if self._db:
            try:
                self._db.close()
            except Exception as close_exc:
                logger.warning(
                    "Task failure: could not close DB session (connection may be dead): %s",
                    close_exc,
                )
            self._db = None

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning("Task retrying", task_id=task_id, error=str(exc))
