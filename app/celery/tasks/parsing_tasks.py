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

# You can either set these via environment variables or pass them directly
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
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=300  # 5 minutes max backoff
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
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    await rate_limiter.acquire()
                    await parsing_service.parse_directory(
                        ParsingRequest(**repo_details),
                        user_id,
                        user_email,
                        project_id,
                        cleanup_graph,
                    )
                    break  # Success, exit loop
                    
                except Exception as e:
                    if "quota exceeded" in str(e).lower() or "429" in str(e):
                        retry_count += 1
                        backoff = rate_limiter.handle_quota_exceeded()
                        logger.warning(
                            f"Rate limit exceeded (attempt {retry_count}/{max_retries}), "
                            f"waiting {backoff}s before retry"
                        )
                        if retry_count < max_retries:
                            await asyncio.sleep(backoff)
                            continue
                    raise  # Re-raise other exceptions or if max retries exceeded

        asyncio.run(run_parsing())
        
    except Exception as e:
        logger.error(f"Error during parsing for project {project_id}: {str(e)}")
        raise

    finally:
        # Ensure rate limiter is cleaned up
        asyncio.run(rate_limiter.shutdown())

logger.info("Parsing tasks module loaded")