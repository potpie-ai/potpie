import asyncio
import logging
from typing import Any, Dict

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService

logger = logging.getLogger(__name__)


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
    bind=True, base=BaseTask, name="app.celery.tasks.parsing_tasks.process_parsing"
)
def process_parsing(
    self, repo_details: Dict[str, Any], user_id: str, user_email: str, project_id: str
) -> None:
    logger.info(f"Task received: Starting parsing process for project {project_id}")
    try:
        parsing_service = ParsingService(self.db)
        asyncio.run(
            parsing_service.parse_directory(
                ParsingRequest(**repo_details), user_id, user_email, project_id
            )
        )
        logger.info(f"Parsing process completed for project {project_id}")
    except Exception as e:
        logger.error(f"Error during parsing for project {project_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


logger.info("Parsing tasks module loaded")
