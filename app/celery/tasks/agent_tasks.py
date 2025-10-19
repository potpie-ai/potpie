import asyncio
import logging
from typing import Optional, List

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.conversations.utils.redis_streaming import RedisStreamManager

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.agent_tasks.execute_agent_background",
)
def execute_agent_background(
    self,
    conversation_id: str,
    run_id: str,
    user_id: str,
    query: str,
    agent_id: str,
    node_ids: Optional[List[str]] = None,
    attachment_ids: List[str] = [],
) -> None:
    """Execute an agent in the background and publish results to Redis streams"""
    redis_manager = RedisStreamManager()

    logger.info(f"Starting background agent execution: {conversation_id}:{run_id}")

    # Set task status to indicate task has started
    redis_manager.set_task_status(conversation_id, run_id, "running")

    try:
        # Execute agent with Redis publishing
        async def run_agent():
            from app.modules.conversations.conversation.conversation_service import (
                ConversationService,
            )
            from app.modules.users.user_service import UserService
            from app.modules.conversations.message.message_model import MessageType
            from app.modules.conversations.message.message_schema import MessageRequest

            # Get user email for service creation
            user_service = UserService(self.db)
            user = user_service.get_user_by_uid(user_id)
            if not user or not user.email:
                raise Exception(f"User email not found for user_id: {user_id}")
            user_email = user.email

            service = ConversationService.create(self.db, user_id, user_email)

            # First, store the user message in history
            message_request = MessageRequest(
                content=query,
                node_ids=node_ids,
                attachment_ids=attachment_ids if attachment_ids else None,
            )

            # Publish start event when actual processing begins
            redis_manager.publish_event(
                conversation_id,
                run_id,
                "start",
                {
                    "agent_id": agent_id or "default",
                    "status": "processing",
                    "message": "Starting message processing",
                },
            )

            # Store the user message and generate AI response
            async for chunk in service.store_message(
                conversation_id,
                message_request,
                MessageType.HUMAN,
                user_id,
                stream=True,
            ):
                # Check for cancellation
                if redis_manager.check_cancellation(conversation_id, run_id):
                    logger.info(
                        f"Agent execution cancelled: {conversation_id}:{run_id}"
                    )
                    redis_manager.publish_event(
                        conversation_id,
                        run_id,
                        "end",
                        {
                            "status": "cancelled",
                            "message": "Generation cancelled by user",
                        },
                    )
                    return

                # Publish chunk event

                # Properly serialize tool calls before sending through Redis
                serialized_tool_calls = []
                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        if hasattr(tool_call, "model_dump"):
                            serialized_tool_calls.append(tool_call.model_dump())
                        elif hasattr(tool_call, "dict"):
                            serialized_tool_calls.append(tool_call.dict())
                        else:
                            # Fallback for already serialized or dict objects
                            serialized_tool_calls.append(tool_call)

                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "chunk",
                    {
                        "content": chunk.message or "",
                        "citations_json": chunk.citations or [],
                        "tool_calls_json": serialized_tool_calls,
                    },
                )

        # Run the async agent execution
        asyncio.run(run_agent())

        # Publish completion event
        redis_manager.publish_event(
            conversation_id,
            run_id,
            "end",
            {"status": "completed", "message": "Agent execution completed"},
        )

        # Set task status to completed
        redis_manager.set_task_status(conversation_id, run_id, "completed")

        logger.info(f"Background agent execution completed: {conversation_id}:{run_id}")

    except Exception as e:
        logger.error(
            f"Background agent execution failed: {conversation_id}:{run_id}: {str(e)}",
            exc_info=True,
        )

        # Set task status to error
        try:
            redis_manager.set_task_status(conversation_id, run_id, "error")
        except Exception as status_error:
            logger.error(f"Failed to set task status to error: {str(status_error)}")

        # Ensure end event is always published
        try:
            redis_manager.publish_event(
                conversation_id, run_id, "end", {"status": "error", "message": str(e)}
            )
        except Exception as redis_error:
            logger.error(f"Failed to publish error event to Redis: {str(redis_error)}")
        raise


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.agent_tasks.execute_regenerate_background",
)
def execute_regenerate_background(
    self,
    conversation_id: str,
    run_id: str,
    user_id: str,
    node_ids: Optional[List[str]] = None,
    attachment_ids: List[str] = [],
) -> None:
    """Execute regeneration in the background and publish results to Redis streams"""
    redis_manager = RedisStreamManager()

    logger.info(f"Starting background regenerate execution: {conversation_id}:{run_id}")

    # Set task status to indicate task has started
    redis_manager.set_task_status(conversation_id, run_id, "running")

    try:
        # Execute regeneration with Redis publishing
        async def run_regeneration():
            from app.modules.conversations.conversation.conversation_service import (
                ConversationService,
            )
            from app.modules.users.user_service import UserService

            # Get user email for service creation
            user_service = UserService(self.db)
            user = user_service.get_user_by_uid(user_id)
            if not user or not user.email:
                raise Exception(f"User email not found for user_id: {user_id}")
            user_email = user.email

            service = ConversationService.create(self.db, user_id, user_email)

            # Publish start event when actual processing begins
            redis_manager.publish_event(
                conversation_id,
                run_id,
                "start",
                {
                    "agent_id": "regenerate",
                    "status": "processing",
                    "message": "Starting regeneration processing",
                },
            )

            # Track if we've received any chunks
            has_chunks = False

            async for chunk in service.regenerate_last_message_background(
                conversation_id, node_ids, attachment_ids
            ):
                has_chunks = True

                # Check for cancellation
                if redis_manager.check_cancellation(conversation_id, run_id):
                    logger.info(
                        f"Regenerate execution cancelled: {conversation_id}:{run_id}"
                    )
                    redis_manager.publish_event(
                        conversation_id,
                        run_id,
                        "end",
                        {
                            "status": "cancelled",
                            "message": "Regeneration cancelled by user",
                        },
                    )
                    return

                # Publish chunk event
                # Properly serialize tool calls before sending through Redis
                serialized_tool_calls = []
                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        if hasattr(tool_call, "model_dump"):
                            serialized_tool_calls.append(tool_call.model_dump())
                        elif hasattr(tool_call, "dict"):
                            serialized_tool_calls.append(tool_call.dict())
                        else:
                            # Fallback for already serialized or dict objects
                            serialized_tool_calls.append(tool_call)

                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "chunk",
                    {
                        "content": chunk.message or "",
                        "citations_json": chunk.citations or [],
                        "tool_calls_json": serialized_tool_calls,
                    },
                )

            # Ensure the database session is properly committed after regeneration
            # The flush_message_buffer should have been called at the end of the generator
            # but we need to ensure the session is committed
            try:
                self.db.commit()
                logger.info(
                    f"Database session committed after regeneration for conversation {conversation_id}"
                )
            except Exception as commit_error:
                logger.error(
                    f"Failed to commit database session after regeneration: {str(commit_error)}"
                )
                self.db.rollback()

            # Log completion of regeneration
            if has_chunks:
                logger.info(
                    f"Regeneration completed successfully for conversation {conversation_id}"
                )
            else:
                logger.warning(
                    f"No chunks received during regeneration for conversation {conversation_id}"
                )

        asyncio.run(run_regeneration())

        # Publish completion event
        redis_manager.publish_event(
            conversation_id,
            run_id,
            "end",
            {"status": "completed", "message": "Regeneration completed"},
        )

        # Set task status to completed
        redis_manager.set_task_status(conversation_id, run_id, "completed")

        logger.info(
            f"Background regenerate execution completed: {conversation_id}:{run_id}"
        )

    except Exception as e:
        logger.error(
            f"Background regenerate execution failed: {conversation_id}:{run_id}: {str(e)}",
            exc_info=True,
        )

        # Set task status to error
        try:
            redis_manager.set_task_status(conversation_id, run_id, "error")
        except Exception as status_error:
            logger.error(f"Failed to set task status to error: {str(status_error)}")

        try:
            redis_manager.publish_event(
                conversation_id, run_id, "end", {"status": "error", "message": str(e)}
            )
        except Exception as redis_error:
            logger.error(f"Failed to publish error event to Redis: {str(redis_error)}")
        raise
