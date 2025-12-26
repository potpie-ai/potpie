from fastapi import APIRouter, Query, HTTPException, Body, Depends
from typing import Optional, List
import logging

from app.modules.intelligence.memory.memory_service_factory import MemoryServiceFactory
from app.modules.auth.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/memories")
async def get_memories(
    user_id: str = Query(..., description="User ID to fetch memories for"),
    project_id: Optional[str] = Query(
        None, description="Optional project ID to scope memories"
    ),
    limit: Optional[int] = Query(
        None, ge=1, le=1000, description="Maximum number of memories to return"
    ),
    user=Depends(AuthService.check_auth),
):
    """
    Get all stored memories/preferences for a specific user.

    Returns all stored user preferences and memories that have been extracted
    from conversation history for the given user. Uses Letta's API to fetch memories.

    - **user_id**: User ID (must match authenticated user)
    - **project_id**: Optional project ID to scope memories
    - **limit**: Maximum number of memories to return
    """
    try:
        # Verify user_id matches authenticated user
        authenticated_user_id = user["user_id"]
        if user_id != authenticated_user_id:
            raise HTTPException(
                status_code=403, detail="Cannot fetch memories for another user"
            )

        memory_service = MemoryServiceFactory.create()

        try:
            search_response = await memory_service.get_all_for_user(
                user_id=user_id, project_id=project_id, limit=limit
            )

            return {
                "memories": [
                    {
                        "memory": result.memory,
                        "metadata": result.metadata,
                        "score": result.score,
                    }
                    for result in search_response.results
                ],
                "total": search_response.total,
                "user_id": user_id,
                "project_id": project_id,
            }
        finally:
            memory_service.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve memories: {str(e)}"
        )


@router.delete("/memories")
async def delete_memories(
    user_id: str = Query(..., description="User ID to delete memories for"),
    project_id: Optional[str] = Query(
        None, description="Optional project ID to scope memories"
    ),
    memory_ids: Optional[List[str]] = Body(
        None,
        description="List of passage IDs to delete. If not provided, deletes all memories for the user/project",
    ),
    user=Depends(AuthService.check_auth),
):
    """
    Delete selected memories from Letta.

    - **user_id**: User ID (must match authenticated user)
    - **project_id**: Optional project ID to scope deletion
    - **memory_ids**: List of passage IDs to delete. If not provided, deletes all memories.

    Returns success status and count of deleted memories.
    """
    try:
        # Verify user_id matches authenticated user
        authenticated_user_id = user["user_id"]
        if user_id != authenticated_user_id:
            raise HTTPException(
                status_code=403, detail="Cannot delete memories for another user"
            )

        memory_service = MemoryServiceFactory.create()

        try:
            success = await memory_service.delete(
                user_id=user_id, project_id=project_id, memory_ids=memory_ids
            )

            if success:
                return {
                    "status": "success",
                    "message": f"Successfully deleted memories for user {user_id}",
                    "user_id": user_id,
                    "project_id": project_id,
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to delete memories")
        finally:
            memory_service.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to delete memories: {str(e)}"
        )
