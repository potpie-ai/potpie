from typing import Any, Dict

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.utils.logger import setup_logger, log_context

logger = setup_logger(__name__)


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
    # Set up logging context with domain IDs
    with log_context(project_id=project_id, user_id=user_id):
        logger.info("Task received: Starting parsing process")
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
                    "Parsing process completed", elapsed_seconds=round(elapsed_time, 2)
                )

            # Use BaseTask's long-lived event loop for consistency
            self.run_async(run_parsing())
        except Exception:
            logger.exception("Error during parsing")
            raise


logger.info("Parsing tasks module loaded")
