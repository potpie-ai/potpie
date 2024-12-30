import asyncio
import logging
from typing import Any, Dict

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

rate_limiter = RateLimiter(name="PARSING")

class BaseTask(Task):
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
    max_retries=3  # Add max retries
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
    try:
        parsing_service = ParsingService(self.db, user_id)

        async def run_parsing():
            try:
                await rate_limiter.acquire()
                await parsing_service.parse_directory(
                    ParsingRequest(**repo_details),
                    user_id,
                    user_email,
                    project_id,
                    cleanup_graph,
                )
            except Exception as e:
                if "quota exceeded" in str(e).lower():
                    rate_limiter.handle_quota_exceeded()
                    # If quota exceeded, retry the whole task
                    logger.warning(f"Rate limit quota exceeded, retrying task for project {project_id}")
                    raise self.retry(
                        exc=e,
                        countdown=60 * (self.request.retries + 1),  # Progressive delay
                        max_retries=3
                    )
                logger.error(f"Error during parsing with rate limiter: {str(e)}")
                raise
            finally:
                logger.debug(f"Rate limiter metrics: {rate_limiter.get_metrics()}")

        asyncio.run(run_parsing())
        logger.info(f"Parsing process completed for project {project_id}")
    except Exception as e:
        logger.error(f"Error during parsing for project {project_id}: {str(e)}")
        raise


logger.info("Parsing tasks module loaded")
