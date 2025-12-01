import logging
from typing import List, Dict, Any, Optional
from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.intelligence.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


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
    Follows the same logging pattern as other Celery background tasks.
    
    Args:
        user_id: User identifier
        messages: List of message dicts with 'role' and 'content' keys
        conversation_id: Optional conversation ID for logging
        project_id: Optional project identifier (will be combined with user_id)
        metadata: Optional metadata to store with memories
    """
    logger.info(
        f"[memory manager ] Starting preference extraction task: user_id={user_id}, "
        f"conversation_id={conversation_id}, project_id={project_id}, messages_count={len(messages)}, task_id={self.request.id}"
    )
    
    try:
        async def run_extraction():
            memory_manager = MemoryManager()
            try:
                result = await memory_manager.extract_preferences_async(
                    messages=messages,
                    user_id=user_id,
                    project_id=project_id,
                    metadata={
                        **(metadata or {}),
                        "conversation_id": conversation_id,
                        "extracted_at": str(self.request.id),
                    }
                )
                # Log what was extracted
                # mem0 returns results in "results" key, not "memories"
                extracted_memories = result.get("results", []) if isinstance(result, dict) else []
                extracted_count = len(extracted_memories) if isinstance(extracted_memories, list) else 0
                extracted_content = []
                if isinstance(extracted_memories, list):
                    for mem in extracted_memories:
                        if isinstance(mem, dict):
                            extracted_content.append(mem.get("memory", mem.get("text", str(mem))))
                        else:
                            extracted_content.append(str(mem))
                
                logger.info(
                    f"[memory manager ] Preference extraction completed successfully: "
                    f"user_id={user_id}, conversation_id={conversation_id}, project_id={project_id}, "
                    f"extracted_count={extracted_count}, extracted_memories={extracted_content}, task_id={self.request.id}"
                )
                return result
            finally:
                memory_manager.close()
        
        # Run async extraction
        self.run_async(run_extraction())
        
        logger.info(
            f"[memory manager ] Preference extraction task finished: user_id={user_id}, "
            f"conversation_id={conversation_id}, project_id={project_id}, task_id={self.request.id}"
        )
    
    except Exception as e:
        logger.error(
            f"[memory manager ] Preference extraction failed: user_id={user_id}, "
            f"conversation_id={conversation_id}, project_id={project_id}, task_id={self.request.id}, error={str(e)}",
            exc_info=True,
        )
        raise

