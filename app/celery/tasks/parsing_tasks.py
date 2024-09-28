import asyncio
from typing import Any, Dict

import redis
from celery.contrib.abortable import AbortableTask
from celery.utils.log import get_task_logger
from redis.exceptions import LockError

from app.celery.celery_app import celery_app, redis_url
from app.core.database import SessionLocal
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService

logger = get_task_logger(__name__)

# Create a Redis client
redis_client = redis.from_url(redis_url)


class BaseTask(AbortableTask):
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.process_parsing",
)
def process_parsing(
    self,
    repo_details: Dict[str, Any],
    user_id: str,
    user_email: str,
    project_id: str,
    cleanup_graph: bool = True,
) -> None:
    logger.info(f"Task received: Starting parsing process for project {project_id}")

    # Acquire a lock for this specific project
    lock_id = f"parsing_lock_{project_id}"
    lock = redis_client.lock(lock_id, timeout=3600)  # Lock expires after 1 hour

    try:
        have_lock = lock.acquire(blocking=False)
        if have_lock:
            try:
                parsing_service = ParsingService(self.db, user_id)

                async def run_parsing():
                    import time

                    start_time = time.time()

                    await parsing_service.parse_directory(
                        ParsingRequest(**repo_details),
                        user_id,
                        user_email,
                        project_id,
                        cleanup_graph,
                    )

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    logger.info(
                        f"Parsing process took {elapsed_time:.2f} seconds for project {project_id}"
                    )

                asyncio.run(run_parsing())
                logger.info(f"Parsing process completed for project {project_id}")
            except Exception as e:
                logger.error(f"Error during parsing for project {project_id}: {str(e)}")
                raise
            finally:
                lock.release()
        else:
            logger.info(
                f"Parsing already in progress for project {project_id}. Skipping."
            )
    except LockError:
        logger.error(f"Failed to acquire lock for project {project_id}")


logger.info("Parsing tasks module loaded")
