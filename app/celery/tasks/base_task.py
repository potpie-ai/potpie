import asyncio
from celery import Task
from contextlib import asynccontextmanager
from app.core.database import SessionLocal
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


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
            self._db.close()
            self._db = None

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning(f"Task {task_id} retrying: {exc}")


class ParseDirectoryTask(BaseTask):
    """
    Specialized task base for parse_directory_unit that handles TimeLimitExceeded.

    When Celery's hard time_limit is exceeded, the worker process is killed with SIGKILL.
    The exception handler in the task code NEVER runs. However, the on_failure callback
    runs in the parent worker process and CAN handle the failure.

    This ensures the Redis completion counter is incremented even on timeout,
    preventing the parsing from getting stuck waiting for the timed-out unit.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure, including TimeLimitExceeded."""
        from billiard.exceptions import TimeLimitExceeded

        # Check if this is a TimeLimitExceeded error
        is_timeout = isinstance(exc, TimeLimitExceeded)

        if is_timeout:
            logger.warning(
                f"Task {task_id} killed by TimeLimitExceeded - handling in on_failure callback"
            )
            self._handle_timeout_failure(task_id, kwargs)

        # Call parent's on_failure for cleanup
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def _handle_timeout_failure(self, task_id: str, kwargs: dict):
        """
        Handle timeout failure by incrementing completion counter and updating DB.

        This is critical because when time_limit is exceeded:
        1. The task's exception handler never runs
        2. The work unit stays in 'pending' or 'processing' state
        3. The Redis counter is never incremented
        4. The system gets stuck waiting forever

        IMPORTANT: DB update and Redis increment are in SEPARATE try blocks.
        A DB failure must NOT prevent the Redis counter from being incremented,
        otherwise the parsing session will be stuck forever.
        """
        from app.celery.coordination import ParsingCoordinator
        from app.modules.parsing.parsing_session_model import ParsingSession
        from app.modules.parsing.parsing_work_unit_model import ParsingWorkUnit
        from datetime import datetime

        work_unit_db_id = kwargs.get('work_unit_db_id')
        project_id = kwargs.get('project_id')
        commit_id = kwargs.get('commit_id')
        user_id = kwargs.get('user_id')
        repo_path = kwargs.get('repo_path')
        work_unit_index = kwargs.get('work_unit_index', 'unknown')

        if not project_id:
            logger.error(f"Cannot handle timeout: missing project_id in kwargs")
            return

        # Create a fresh DB session for this callback
        db = SessionLocal()

        # Step 1: Try to update work unit status (separate try block - failure here must not block counter)
        try:
            if work_unit_db_id:
                work_unit = db.query(ParsingWorkUnit).filter(
                    ParsingWorkUnit.id == work_unit_db_id
                ).first()
                if work_unit:
                    work_unit.status = 'failed'
                    work_unit.attempt_count = (work_unit.attempt_count or 0) + 1
                    work_unit.error_message = 'Task killed by TimeLimitExceeded (hard timeout)'
                    work_unit.last_error_at = datetime.utcnow()
                    db.commit()
                    logger.info(
                        f"[Unit {work_unit_index}] Marked work unit as failed due to timeout"
                    )
        except Exception as db_error:
            logger.error(
                f"[Unit {work_unit_index}] Failed to update work unit status: {db_error}. "
                f"Continuing to increment Redis counter anyway."
            )
            try:
                db.rollback()
            except Exception:
                pass

        # Step 2: ALWAYS try to increment Redis counter and trigger finalization
        # This is in a separate try block so DB failures above don't prevent counter update
        try:
            # Find the parsing session
            session_query = db.query(ParsingSession).filter(
                ParsingSession.project_id == project_id,
                ParsingSession.completed_at.is_(None)
            )

            if commit_id is not None:
                session_query = session_query.filter(
                    ParsingSession.commit_id == commit_id
                )
            else:
                session_query = session_query.filter(
                    ParsingSession.commit_id.is_(None)
                )

            session = session_query.first()

            if session:
                # Increment Redis counter
                redis_client = self.app.backend.client
                completed_count, is_last = ParsingCoordinator.increment_completed(
                    redis_client,
                    project_id,
                    commit_id,
                    session.total_work_units,
                    work_unit_id=str(work_unit_db_id) if work_unit_db_id else None
                )

                logger.info(
                    f"[Unit {work_unit_index}] Timeout failure counted: "
                    f"{completed_count}/{session.total_work_units}"
                )

                # If this was the last unit (even though it timed out), trigger finalization
                if is_last:
                    logger.warning(
                        f"[Unit {work_unit_index}] Last worker (timeout) - triggering finalization"
                    )
                    # Import here to avoid circular import
                    from app.celery.tasks.parsing_tasks import finalize_parsing
                    finalize_parsing.apply_async(
                        kwargs={
                            'project_id': project_id,
                            'user_id': user_id,
                            'repo_path': repo_path,
                            'commit_id': commit_id
                        },
                        countdown=5
                    )
            else:
                logger.warning(
                    f"[Unit {work_unit_index}] No active parsing session found for timeout handling"
                )

        except Exception as e:
            logger.exception(f"[Unit {work_unit_index}] Error incrementing completion counter: {e}")
        finally:
            try:
                db.close()
            except Exception:
                pass


class InferenceTask(BaseTask):
    """
    Specialized task base for run_inference_unit that handles TimeLimitExceeded.

    Same pattern as ParseDirectoryTask - ensures Redis counter is incremented
    even when worker is killed by SIGKILL (OOM, hard limit, pod eviction).
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure, including TimeLimitExceeded."""
        from billiard.exceptions import TimeLimitExceeded

        # Check if this is a TimeLimitExceeded error
        is_timeout = isinstance(exc, TimeLimitExceeded)

        if is_timeout:
            logger.warning(
                f"[Inference] Task {task_id} killed by TimeLimitExceeded - handling in on_failure callback"
            )
            self._handle_timeout_failure(task_id, kwargs)

        # Call parent's on_failure for cleanup
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def _handle_timeout_failure(self, task_id: str, kwargs: dict):
        """
        Handle timeout failure by incrementing completion counter and updating DB.

        IMPORTANT: DB update and Redis increment are in SEPARATE try blocks.
        A DB failure must NOT prevent the Redis counter from being incremented,
        otherwise the inference session will be stuck forever.
        """
        from app.celery.coordination import InferenceCoordinator
        from app.modules.parsing.inference_session_model import InferenceSession
        from app.modules.parsing.inference_work_unit_model import InferenceWorkUnit
        from datetime import datetime
        from uuid import UUID

        work_unit_id = kwargs.get('work_unit_id')
        session_id = kwargs.get('session_id')
        project_id = kwargs.get('project_id')
        user_id = kwargs.get('user_id')
        directory_path = kwargs.get('directory_path', 'unknown')

        if not project_id or not session_id:
            logger.error(
                f"[Inference] Cannot handle timeout: missing project_id or session_id in kwargs"
            )
            return

        # Create a fresh DB session for this callback
        db = SessionLocal()

        # Step 1: Try to update work unit status (separate try block - failure must not block counter)
        try:
            if work_unit_id:
                work_unit = db.query(InferenceWorkUnit).filter(
                    InferenceWorkUnit.id == UUID(work_unit_id)
                ).first()
                if work_unit:
                    work_unit.status = 'failed'
                    work_unit.attempt_count = (work_unit.attempt_count or 0) + 1
                    work_unit.error_message = 'Task killed by TimeLimitExceeded (hard timeout)'
                    work_unit.last_error_at = datetime.utcnow()
                    db.commit()
                    logger.info(
                        f"[Inference {directory_path}] Marked work unit as failed due to timeout"
                    )
        except Exception as db_error:
            logger.error(
                f"[Inference {directory_path}] Failed to update work unit status: {db_error}. "
                f"Continuing to increment Redis counter anyway."
            )
            try:
                db.rollback()
            except Exception:
                pass

        # Step 2: ALWAYS try to increment Redis counter and trigger finalization
        try:
            # Find the inference session
            inference_session = db.query(InferenceSession).filter(
                InferenceSession.id == UUID(session_id)
            ).first()

            if inference_session:
                # Increment Redis counter
                redis_client = self.app.backend.client
                completed_count, is_last = InferenceCoordinator.increment_completed(
                    redis_client,
                    project_id,
                    session_id,
                    inference_session.total_work_units,
                    work_unit_id=work_unit_id
                )

                logger.info(
                    f"[Inference {directory_path}] Timeout failure counted: "
                    f"{completed_count}/{inference_session.total_work_units}"
                )

                # If this was the last unit, trigger finalization
                if is_last:
                    logger.warning(
                        f"[Inference {directory_path}] Last worker (timeout) - triggering finalization"
                    )
                    from app.celery.tasks.parsing_tasks import finalize_project_after_inference
                    finalize_project_after_inference.apply_async(
                        kwargs={
                            'project_id': project_id,
                            'user_id': user_id,
                            'session_id': session_id,
                        },
                        countdown=5
                    )
            else:
                logger.warning(
                    f"[Inference {directory_path}] No inference session found for timeout handling"
                )

        except Exception as e:
            logger.exception(f"[Inference {directory_path}] Error incrementing completion counter: {e}")
        finally:
            try:
                db.close()
            except Exception:
                pass
