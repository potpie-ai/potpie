from typing import Any, Dict, List, Optional

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.intelligence.memory.memory_service_factory import MemoryServiceFactory
from app.modules.utils.logger import log_context, setup_logger

logger = setup_logger(__name__)


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.memory_tasks.extract_user_preferences",
)
def extract_user_preferences(
    self,
    user_id: str,
    messages: List[Dict[str, str]],
    conversation_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Extract user preferences from conversation messages asynchronously.
    """
    request_id = getattr(self.request, "id", None)

    with log_context(
        user_id=user_id,
        conversation_id=conversation_id,
        project_id=project_id,
        task_id=request_id,
    ):
        logger.info(
            "[memory manager] Starting preference extraction task",
            messages_count=len(messages),
        )

        try:

            async def run_extraction():
                memory_service = MemoryServiceFactory.create()
                try:
                    result = await memory_service.add(
                        messages=messages,
                        user_id=user_id,
                        project_id=project_id,
                        metadata={
                            **(metadata or {}),
                            "conversation_id": conversation_id,
                            "extracted_at": str(request_id),
                        },
                    )

                    extracted_memories = (
                        result.get("results", []) if isinstance(result, dict) else []
                    )
                    extracted_count = (
                        len(extracted_memories)
                        if isinstance(extracted_memories, list)
                        else 0
                    )
                    extracted_content: List[str] = []
                    if isinstance(extracted_memories, list):
                        for mem in extracted_memories:
                            if isinstance(mem, dict):
                                extracted_content.append(
                                    mem.get("memory", mem.get("text", str(mem)))
                                )
                            else:
                                extracted_content.append(str(mem))

                    logger.info(
                        "[memory manager] Preference extraction completed successfully",
                        extracted_count=extracted_count,
                        extracted_memories=extracted_content,
                    )
                    return result
                finally:
                    memory_service.close()

            # Run async extraction
            self.run_async(run_extraction())

            logger.info("[memory manager] Preference extraction task finished")

        except Exception as exc:
            logger.exception(
                "[memory manager] Preference extraction failed",
                error=str(exc),
            )
            raise
