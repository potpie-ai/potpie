"""
Gap Analysis Router - API endpoints for triggering gap analysis tasks.

This router provides endpoints to:
- Start a gap analysis task for a project
- Poll for task status
- Stream results via Redis streams
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.celery.tasks.agent_tasks import execute_gap_analysis_background
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.conversations.utils.conversation_routing import (
    ensure_unique_run_id,
    normalize_run_id,
    redis_stream_generator,
)
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.usage.usage_service import UsageService

logger = logging.getLogger(__name__)

router = APIRouter()


class GapAnalysisRequest(BaseModel):
    """Request model for starting a gap analysis task"""

    query: str = Field(..., description="User's request for gap analysis")
    project_id: str = Field(..., description="Project ID to analyze")
    conversation_id: Optional[str] = Field(
        None, description="Existing conversation ID (optional)"
    )
    node_ids: Optional[List[str]] = Field(
        None, description="Specific nodes to focus on"
    )
    attachment_ids: List[str] = Field(
        default_factory=list, description="File attachments"
    )


class GapAnalysisStartResponse(BaseModel):
    """Response model for starting a gap analysis task"""

    run_id: str = Field(..., description="Unique run ID for tracking")
    conversation_id: str = Field(..., description="Conversation ID")
    status: str = Field(..., description="Initial status")


class GapAnalysisStatusResponse(BaseModel):
    """Response model for gap analysis status"""

    run_id: str = Field(..., description="Run ID")
    conversation_id: str = Field(..., description="Conversation ID")
    status: str = Field(..., description="Current status")


@router.post("/gap-analysis", response_model=GapAnalysisStartResponse)
async def create_gap_analysis(
    request: GapAnalysisRequest,
    stream: bool = Query(True, description="Whether to stream the response"),
    session_id: Optional[str] = Query(None, description="Session ID for reconnection"),
    prev_human_message_id: Optional[str] = Query(
        None, description="Previous human message ID for deterministic session ID"
    ),
    cursor: Optional[str] = Query(None, description="Stream cursor for replay"),
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    """
    Start a gap analysis task for a specific project.

    This endpoint creates a background task that performs JANUS-style gap analysis
    contextualized to the specified project:

    **Request Parameters:**
    - `query`: User's request (e.g., "Add user authentication with OAuth")
    - `project_id`: Project to analyze (determines which codebase to explore)
    - `conversation_id`: Optional existing conversation to continue
    - `stream`: Whether to stream the response (default: True)

    **Gap Analysis Process:**
    1. Classifies work intent (Refactoring, Build, Architecture, etc.)
    2. Explores the project's codebase using:
       - Knowledge graph queries for project-specific patterns
       - fetch_file for codebase search
       - External research for best practices
    3. Identifies ambiguities and gaps in the request
    4. Generates 5-15 prioritized MCQs with:
       - Multiple choice options (3-5 per question)
       - Answer recommendations based on project context
       - Context references to actual project files

    **Response (non-streaming):**
    - Returns immediately with `run_id` and `conversation_id`
    - Use `run_id` to poll for status

    **Response (streaming):**
    - Returns a streaming response with events as they happen

    **Example:**
    ```json
    {
      "query": "Add real-time notifications using WebSockets",
      "project_id": "my-saas-app-123"
    }
    ```
    """
    user_id = user["user_id"]

    # Check usage limit
    checked = await UsageService.check_usage_limit(user_id)
    if not checked:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Subscription required to use gap analysis.",
        )

    redis_manager = RedisStreamManager()

    try:
        # Use existing conversation or create new one
        if request.conversation_id:
            conversation_id = request.conversation_id
        else:
            conversation_id = str(uuid.uuid4())

        # Generate run_id
        run_id = normalize_run_id(
            conversation_id, user_id, session_id, prev_human_message_id
        )

        # For fresh requests without cursor, ensure we get a unique stream
        if not cursor:
            run_id = ensure_unique_run_id(conversation_id, run_id)

        # Set initial "queued" status before starting the task
        redis_manager.set_task_status(conversation_id, run_id, "queued")

        # Publish a queued event so the client knows the task is accepted
        redis_manager.publish_event(
            conversation_id,
            run_id,
            "queued",
            {
                "status": "queued",
                "message": "Gap analysis task queued for processing",
            },
        )

        # Trigger Celery task with project context
        task_result = execute_gap_analysis_background.delay(
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            query=request.query,
            project_id=request.project_id,
            node_ids=request.node_ids,
            attachment_ids=request.attachment_ids,
        )

        # Store the Celery task ID for later revocation
        redis_manager.set_task_id(conversation_id, run_id, task_result.id)

        logger.info(
            f"Gap analysis task created: run_id={run_id}, "
            f"task_id={task_result.id}, "
            f"project_id={request.project_id}, "
            f"conversation_id={conversation_id}, user={user_id}"
        )

        if stream:
            # Wait for background task to start (with health check)
            task_started = redis_manager.wait_for_task_start(
                conversation_id, run_id, timeout=30
            )

            if not task_started:
                logger.warning(
                    f"Gap analysis task failed to start within 30s for {conversation_id}:{run_id} - may still be queued"
                )

            # Return Redis stream response
            return StreamingResponse(
                redis_stream_generator(conversation_id, run_id, cursor),
                media_type="text/event-stream",
            )

        # Non-streaming: return immediately with run_id for polling
        return GapAnalysisStartResponse(
            run_id=run_id,
            conversation_id=conversation_id,
            status="pending",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create gap analysis task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create gap analysis task: {str(e)}",
        )


@router.get("/gap-analysis/{run_id}/status", response_model=GapAnalysisStatusResponse)
async def get_gap_analysis_status(
    run_id: str,
    conversation_id: str = Query(..., description="Conversation ID"),
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    """
    Get the status of a gap analysis task.

    **Status Values:**
    - `queued`: Task is waiting to be processed
    - `running`: Task is currently being processed
    - `completed`: Task finished successfully
    - `cancelled`: Task was cancelled by user
    - `error`: Task failed with an error
    """
    redis_manager = RedisStreamManager()

    try:
        task_status = redis_manager.get_task_status(conversation_id, run_id)

        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {run_id}",
            )

        return GapAnalysisStatusResponse(
            run_id=run_id,
            conversation_id=conversation_id,
            status=task_status,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}",
        )


@router.post("/gap-analysis/{run_id}/cancel")
async def cancel_gap_analysis(
    run_id: str,
    conversation_id: str = Query(..., description="Conversation ID"),
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    """
    Cancel a running gap analysis task.

    This will signal the task to stop at the next cancellation check point.
    """
    redis_manager = RedisStreamManager()

    try:
        # Check if task exists
        task_status = redis_manager.get_task_status(conversation_id, run_id)

        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {run_id}",
            )

        if task_status in ["completed", "cancelled", "error"]:
            return {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "status": task_status,
                "message": f"Task already {task_status}",
            }

        # Set cancellation flag
        redis_manager.request_cancellation(conversation_id, run_id)

        logger.info(f"Cancellation requested for gap analysis: {conversation_id}:{run_id}")

        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "status": "cancelling",
            "message": "Cancellation requested",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}",
        )
