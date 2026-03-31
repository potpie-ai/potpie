from pathlib import Path
from typing import Any, Dict

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.utils.colgrep_index import build_colgrep_index
from app.modules.utils.logger import setup_logger, log_context

logger = setup_logger(__name__)


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.process_colgrep_index",
)
def process_colgrep_index(
    self,
    repo_root: str,
    repos_base_path: str | None = None,
) -> None:
    with log_context(repo_root=repo_root):
        logger.info("Task received: Starting ColGREP index build")
        base_path = Path(repos_base_path) if repos_base_path else None
        build_colgrep_index(repo_root, base_path)
        logger.info("ColGREP index build task completed")


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
        parsing_service = None
        try:
            parsing_service = ParsingService(
                self.db, user_id, raise_library_exceptions=True
            )

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

            # Run parsing in a fresh event loop (asyncio.run)
            self.run_async(run_parsing())
        except Exception:
            logger.exception("Error during parsing")
            raise
        finally:
            if parsing_service is not None:
                try:
                    parsing_service.close()
                except Exception:
                    pass


logger.info("Parsing tasks module loaded")
