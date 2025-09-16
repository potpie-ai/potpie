import asyncio
import logging
from typing import Any, Dict

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService

logger = logging.getLogger(__name__)


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


logger.info("Parsing tasks module loaded")
