from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging

from app.modules.intelligence.memory.memory_manager import MemoryManager
from app.modules.intelligence.memory.memory_service_factory import MemoryServiceFactory

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/memories")
async def get_memories(
    user_id: str = Query(..., description="User ID to fetch memories for"),
    project_id: Optional[str] = Query(None, description="Optional project ID to scope memories"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of memories to return"),
):
    """
    Get all stored memories/preferences for a specific user.
    
    Returns all stored user preferences and memories that have been extracted
    from conversation history for the given user. Uses mem0's API to fetch memories.
    """
    try:
        memory_service = MemoryServiceFactory.create()
        memory_manager = MemoryManager(memory_service=memory_service)
        
        try:
            result = await memory_manager.get_all_memories(
                user_id=user_id,
                project_id=project_id,
                limit=limit
            )
            return {
                "memories": result["memories"],
                "total": result["total"],
                "user_id": user_id,
                "project_id": project_id
            }
        finally:
            memory_manager.close()
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memories: {str(e)}"
        )

