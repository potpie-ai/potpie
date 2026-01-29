"""
Shared utilities for conversation routing endpoints (v1 and v2).
Contains common functions for session management, Redis streaming, and Celery task execution.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.modules.conversations.conversation.conversation_schema import (
    ChatMessageResponse,
)
from app.modules.conversations.utils.redis_streaming import RedisStreamManager

logger = logging.getLogger(__name__)


def normalize_run_id(
    conversation_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    prev_human_message_id: Optional[str] = None,
) -> str:
    """
    Generate user-scoped deterministic session IDs.
    Format: conversation:{user_id}:{prev_human_message_id}
    If no prev_human_message_id provided, defaults to 'new'
    """
    if session_id:
        return session_id

    # Use provided prev_human_message_id or default to 'new'
    message_id = prev_human_message_id if prev_human_message_id else "new"
    return f"conversation:{user_id}:{message_id}"


def ensure_unique_run_id(conversation_id: str, run_id: str) -> str:
    """
    Ensure the run_id is unique by checking if a stream already exists.
    If it exists, append a counter to make it unique.
    """
    redis_manager = RedisStreamManager()
    original_run_id = run_id
    counter = 1

    # Find a unique run_id if the original already has an active stream
    while redis_manager.redis_client.exists(
        redis_manager.stream_key(conversation_id, run_id)
    ):
        run_id = f"{original_run_id}-{counter}"
        counter += 1

    return run_id


def redis_stream_generator(
    conversation_id: str, run_id: str, cursor: Optional[str] = None
) -> Generator[str, None, None]:
    """Stream events from Redis to client"""

    def json_serializer(obj):
        """Custom JSON serializer to handle bytes objects"""
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return str(obj)

    redis_manager = RedisStreamManager()

    try:
        for event in redis_manager.consume_stream(conversation_id, run_id, cursor):
            # Convert to ChatMessageResponse format for compatibility
            if event.get("type") == "chunk":
                tool_calls = event.get("tool_calls", [])
                content = event.get("content", "")
                response = ChatMessageResponse(
                    message=content,
                    citations=event.get("citations", []),
                    tool_calls=tool_calls,
                )
                json_response = json.dumps(response.dict(), default=json_serializer)
                yield json_response

            elif event.get("type") == "queued":
                # Send a queued status to the client
                response = ChatMessageResponse(
                    message="",
                    citations=[],
                    tool_calls=[],
                )
                json_response = json.dumps(response.dict(), default=json_serializer)
                yield json_response

            elif event.get("type") == "end":
                # End the stream when we receive an end event
                break

    except Exception as e:
        logger.error(f"Redis streaming error: {str(e)}")
        # Don't yield error events to match original behavior


def start_celery_task_and_stream(
    conversation_id: str,
    run_id: str,
    user_id: str,
    query: str,
    agent_id: Optional[str],
    node_ids: list,
    attachment_ids: list,
    cursor: Optional[str] = None,
) -> StreamingResponse:
    """
    Start a Celery background task and return a streaming response.
    This is the common pattern used by both v1 and v2 endpoints.
    """
    from app.celery.tasks.agent_tasks import execute_agent_background

    redis_manager = RedisStreamManager()

    # Set initial "queued" status before starting the task
    redis_manager.set_task_status(conversation_id, run_id, "queued")

    # Publish a queued event so the client knows the task is accepted
    redis_manager.publish_event(
        conversation_id,
        run_id,
        "queued",
        {
            "status": "queued",
            "message": "Task queued for processing",
        },
    )

    # Start background task
    task_result = execute_agent_background.delay(
        conversation_id=conversation_id,
        run_id=run_id,
        user_id=user_id,
        query=query,
        agent_id=agent_id,
        node_ids=node_ids,
        attachment_ids=attachment_ids or [],
    )

    # Store the Celery task ID for later revocation
    redis_manager.set_task_id(conversation_id, run_id, task_result.id)
    logger.info(f"Started agent task {task_result.id} for {conversation_id}:{run_id}")

    # Wait for background task to start (with health check)
    # Increased timeout to 30 seconds to handle queued tasks
    task_started = redis_manager.wait_for_task_start(
        conversation_id, run_id, timeout=30
    )

    if not task_started:
        logger.warning(
            f"Background task failed to start within 30s for {conversation_id}:{run_id} - may still be queued"
        )
        # Don't fail - the stream consumer will wait up to 120 seconds

    # Return Redis stream response
    return StreamingResponse(
        redis_stream_generator(conversation_id, run_id, cursor),
        media_type="text/event-stream",
    )


async def start_celery_task_and_wait(
    conversation_id: str,
    run_id: str,
    user_id: str,
    query: str,
    agent_id: Optional[str],
    node_ids: list,
    attachment_ids: list,
) -> ChatMessageResponse:
    """
    Start a Celery background task and wait for the complete response.
    Collects all chunks from Redis stream and returns the full response.
    Used for non-streaming requests in v2 API.
    """
    from app.celery.tasks.agent_tasks import execute_agent_background

    redis_manager = RedisStreamManager()

    # Set initial "queued" status before starting the task
    redis_manager.set_task_status(conversation_id, run_id, "queued")

    # Publish a queued event so the client knows the task is accepted
    redis_manager.publish_event(
        conversation_id,
        run_id,
        "queued",
        {
            "status": "queued",
            "message": "Task queued for processing",
        },
    )

    # Start background task
    task_result = execute_agent_background.delay(
        conversation_id=conversation_id,
        run_id=run_id,
        user_id=user_id,
        query=query,
        agent_id=agent_id,
        node_ids=node_ids,
        attachment_ids=attachment_ids or [],
    )

    # Store the Celery task ID for later revocation
    redis_manager.set_task_id(conversation_id, run_id, task_result.id)
    logger.info(
        f"Started agent task {task_result.id} for {conversation_id}:{run_id} (non-streaming)"
    )

    # Wait for background task to start (with health check)
    task_started = redis_manager.wait_for_task_start(
        conversation_id, run_id, timeout=30
    )

    if not task_started:
        logger.warning(
            f"Background task failed to start within 30s for {conversation_id}:{run_id} - may still be queued"
        )

    # Collect all chunks from the stream
    # Run the synchronous stream consumption in a thread pool to avoid blocking
    full_message = ""
    all_citations = []
    all_tool_calls = []
    error_message = None

    def collect_from_stream():
        """Synchronous function to collect all events from Redis stream"""
        events = []
        try:
            for event in redis_manager.consume_stream(conversation_id, run_id):
                events.append(event)
                # Stop collecting when we get an end event
                if event.get("type") == "end":
                    break
        except Exception as e:
            logger.error(
                f"Error consuming Redis stream for {conversation_id}:{run_id}: {str(e)}",
                exc_info=True,
            )
            # Add error event to signal failure
            events.append(
                {
                    "type": "end",
                    "status": "error",
                    "message": f"Stream error: {str(e)}",
                }
            )
        return events

    try:
        # Run stream consumption in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            events = await loop.run_in_executor(executor, collect_from_stream)

        # Process collected events
        for event in events:
            if event.get("type") == "chunk":
                # Accumulate message content
                content = event.get("content", "")
                if content:
                    full_message += content

                # Accumulate citations (avoid duplicates)
                citations = event.get("citations", [])
                if citations:
                    # Merge citations, avoiding duplicates
                    for citation in citations:
                        if citation not in all_citations:
                            all_citations.append(citation)

                # Accumulate tool calls
                tool_calls = event.get("tool_calls", [])
                if tool_calls:
                    all_tool_calls.extend(tool_calls)

            elif event.get("type") == "end":
                status = event.get("status", "completed")
                if status == "error":
                    error_message = event.get("message", "Unknown error occurred")
                    logger.error(
                        f"Task completed with error for {conversation_id}:{run_id}: {error_message}"
                    )
                elif status == "cancelled":
                    error_message = "Task was cancelled"
                    logger.info(f"Task cancelled for {conversation_id}:{run_id}")
                break

        # If we got an error, raise an exception
        if error_message:
            raise HTTPException(status_code=500, detail=error_message)

        # Return the complete response
        return ChatMessageResponse(
            message=full_message,
            citations=all_citations,
            tool_calls=all_tool_calls,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Error collecting response from Redis stream for {conversation_id}:{run_id}: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to collect response: {str(e)}"
        )
