import asyncio
import os
from typing import Optional, List

from sqlalchemy.orm import Session

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.users.user_model import User
from app.modules.users.user_service import UserService
from app.modules.utils.logger import setup_logger, log_context

logger = setup_logger(__name__)


def _resolve_user_email_for_celery(db: Session, user_id: str) -> str:
    """
    Resolve user email using sync DB. Call from Celery task body before run_async();
    do not call from inside the coroutine.
    """
    user = UserService(db).get_user_by_uid(user_id)
    if not user:
        direct_user = db.query(User).filter(User.uid == user_id).first()
        if direct_user:
            logger.warning(
                "UserService.get_user_by_uid returned None but direct query found user: %s, email: %s",
                direct_user.uid,
                direct_user.email,
            )
            user = direct_user
        else:
            logger.warning(
                "User not found in database for user_id: %s. Using empty string as fallback.",
                user_id,
            )
            return ""
    email = getattr(user, "email", None) or ""
    if not email:
        logger.warning(
            "User found but email is None/empty for user_id: %s, user.uid: %s, email value: %r. Using empty string as fallback.",
            user_id,
            getattr(user, "uid", "N/A"),
            getattr(user, "email", "N/A"),
        )
        return ""
    logger.debug("Retrieved user email for user_id: %s", user_id)
    return email


def _clear_pydantic_ai_http_client_cache() -> None:
    """Clear pydantic_ai's globally cached async HTTP client(s).

    In Celery workers we use asyncio.run() per task, so each task has a new event loop.
    The cached httpx.AsyncClient is created on first use and tied to that task's loop.
    Reusing it in a later task (different loop, previous loop closed) can cause the
    model request stream to hang or yield no chunks. Clearing the cache at task start
    forces the next model creation to get a fresh client bound to the current loop.
    """
    if os.getenv("CELERY_WORKER") != "1":
        return
    try:
        from pydantic_ai.models import _cached_async_http_client

        _cached_async_http_client.cache_clear()
    except Exception as e:
        logger.debug("Could not clear pydantic_ai HTTP client cache: %s", e)


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
    local_mode: bool = False,
    tunnel_url: Optional[str] = None,
) -> None:
    """Execute an agent in the background and publish results to Redis streams"""
    # Clear pydantic_ai's globally cached async HTTP client so this task gets a fresh
    # client bound to this task's event loop. With asyncio.run() per task, the previous
    # task's loop is closed, so reusing the cached client would use a client tied to a
    # closed loop and can hang or yield no chunks (e.g. model request stream stuck).
    _clear_pydantic_ai_http_client_cache()

    redis_manager = RedisStreamManager()

    # Set up logging context with domain IDs
    with log_context(conversation_id=conversation_id, user_id=user_id, run_id=run_id):
        logger.info(
            f"Starting background agent execution with tunnel_url={tunnel_url}, "
            f"local_mode={local_mode}, conversation_id={conversation_id}"
        )

        # Set task status to indicate task has started
        try:
            redis_manager.set_task_status(conversation_id, run_id, "running")
            logger.info("Task status set to running (Redis ok)")
        except Exception as e:
            logger.warning("Failed to set task status in Redis", error=str(e))

        try:
            user_email = _resolve_user_email_for_celery(self.db, user_id)

            # Execute agent with Redis publishing
            async def run_agent():
                import asyncio

                from app.modules.conversations.conversation.conversation_service import (
                    ConversationService,
                )
                from app.modules.conversations.exceptions import GenerationCancelled
                from app.modules.conversations.message.message_model import MessageType
                from app.modules.conversations.message.message_schema import (
                    MessageRequest,
                )
                from app.modules.conversations.conversation.conversation_store import (
                    ConversationStore,
                )
                from app.modules.conversations.message.message_store import MessageStore

                logger.info("run_agent: coroutine started, acquiring async_db")
                async with self.async_db() as async_db:
                    logger.info(
                        "run_agent: async_db acquired, creating ConversationService"
                    )
                    conversation_store = ConversationStore(self.db, async_db)
                    message_store = MessageStore(self.db, async_db)

                    service = ConversationService.create(
                        conversation_store=conversation_store,
                        message_store=message_store,
                        db=self.db,
                        user_id=user_id,
                        user_email=user_email,
                    )

                    # First, store the user message in history
                    message_request = MessageRequest(
                        content=query,
                        node_ids=node_ids,
                        attachment_ids=attachment_ids if attachment_ids else None,
                        tunnel_url=tunnel_url,
                    )

                    # Publish start event when actual processing begins
                    logger.info(
                        "run_agent: publishing start event, calling store_message"
                    )
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

                    # Store the user message and generate AI response (pass cancellation check so agent can stop cooperatively)
                    check_cancelled = lambda: redis_manager.check_cancellation(
                        conversation_id, run_id
                    )
                    try:
                        async for chunk in service.store_message(
                            conversation_id,
                            message_request,
                            MessageType.HUMAN,
                            user_id,
                            stream=True,
                            local_mode=local_mode,
                            run_id=run_id,
                            check_cancelled=check_cancelled,
                        ):
                            # Check for cancellation (redundant with cooperative check in agent, but keeps early exit)
                            if redis_manager.check_cancellation(
                                conversation_id, run_id
                            ):
                                logger.info("Agent execution cancelled")
                                # Do not flush here - stop_generation saves from Redis snapshot to avoid duplicates
                                redis_manager.publish_event(
                                    conversation_id,
                                    run_id,
                                    "end",
                                    {
                                        "status": "cancelled",
                                        "message": "Generation cancelled by user",
                                    },
                                )
                                return False  # Indicate cancellation

                            # Publish chunk event (run in executor to avoid blocking the event loop)
                            serialized_tool_calls = []
                            if chunk.tool_calls:
                                for tool_call in chunk.tool_calls:
                                    if hasattr(tool_call, "model_dump"):
                                        serialized_tool_calls.append(
                                            tool_call.model_dump()
                                        )
                                    elif hasattr(tool_call, "dict"):
                                        serialized_tool_calls.append(tool_call.dict())
                                    else:
                                        serialized_tool_calls.append(tool_call)

                            payload = {
                                "content": chunk.message or "",
                                "citations_json": chunk.citations or [],
                                "tool_calls_json": serialized_tool_calls,
                                "thinking": chunk.thinking,
                            }

                            def _publish_chunk():
                                redis_manager.publish_event(
                                    conversation_id, run_id, "chunk", payload
                                )

                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, _publish_chunk)

                        return True  # Indicate successful completion (loop finished)
                    except GenerationCancelled:
                        logger.info("Agent execution cancelled (GenerationCancelled)")
                        # Do not flush here - stop_generation saves from Redis snapshot to avoid duplicates
                        redis_manager.publish_event(
                            conversation_id,
                            run_id,
                            "end",
                            {
                                "status": "cancelled",
                                "message": "Generation cancelled by user",
                            },
                        )
                        return False  # Indicate cancellation

                    return True  # Indicate successful completion

            # Run the async agent execution in a fresh event loop (asyncio.run).
            # Convert asyncio.CancelledError to RuntimeError so Celery's result callback
            # receives (failed, retval, runtime) instead of ExceptionInfo (avoids
            # "cannot unpack non-iterable ExceptionInfo object").
            logger.info("Entering run_async (agent coroutine)")
            try:
                completed = self.run_async(run_agent())
            except asyncio.CancelledError as e:
                logger.warning(
                    "Agent run was cancelled (asyncio.CancelledError); "
                    "re-raising as RuntimeError for Celery"
                )
                raise RuntimeError("Agent stream was cancelled during execution") from e

            # Only publish completion event if not cancelled
            if completed:
                # Publish completion event
                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "end",
                    {"status": "completed", "message": "Agent execution completed"},
                )

                # Set task status to completed
                redis_manager.set_task_status(conversation_id, run_id, "completed")

                logger.info("Background agent execution completed")
            else:
                logger.info("Background agent execution cancelled")

            # Return the completion status so on_success can check if it was cancelled
            return completed

        except Exception as e:
            logger.exception(
                "Background agent execution failed",
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
            )

            # Set task status to error
            try:
                redis_manager.set_task_status(conversation_id, run_id, "error")
            except Exception:
                logger.exception(
                    "Failed to set task status to error",
                    conversation_id=conversation_id,
                    run_id=run_id,
                )

            # Ensure end event is always published
            try:
                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "end",
                    {"status": "error", "message": str(e)},
                )
            except Exception:
                logger.exception(
                    "Failed to publish error event to Redis",
                    conversation_id=conversation_id,
                    run_id=run_id,
                )
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
    local_mode: bool = False,
) -> None:
    """Execute regeneration in the background and publish results to Redis streams"""
    _clear_pydantic_ai_http_client_cache()

    redis_manager = RedisStreamManager()

    # Set up logging context with domain IDs
    with log_context(conversation_id=conversation_id, user_id=user_id, run_id=run_id):
        logger.info("Starting background regenerate execution")

    # Set task status to indicate task has started
    try:
        redis_manager.set_task_status(conversation_id, run_id, "running")
        logger.info("Regenerate task status set to running (Redis ok)")
    except Exception as e:
        logger.warning("Failed to set task status in Redis", error=str(e))

    try:
        user_email = _resolve_user_email_for_celery(self.db, user_id)

        # Execute regeneration with Redis publishing
        async def run_regeneration():
            from app.modules.conversations.conversation.conversation_service import (
                ConversationService,
            )
            from app.modules.conversations.exceptions import GenerationCancelled
            from app.modules.conversations.conversation.conversation_store import (
                ConversationStore,
            )
            from app.modules.conversations.message.message_store import MessageStore

            async with self.async_db() as async_db:
                conversation_store = ConversationStore(self.db, async_db)
                message_store = MessageStore(self.db, async_db)

                service = ConversationService.create(
                    conversation_store=conversation_store,
                    message_store=message_store,
                    db=self.db,
                    user_id=user_id,
                    user_email=user_email,
                )
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
                check_cancelled = lambda: redis_manager.check_cancellation(
                    conversation_id, run_id
                )
                try:
                    async for chunk in service.regenerate_last_message_background(
                        conversation_id,
                        node_ids,
                        attachment_ids,
                        local_mode=local_mode,
                        run_id=run_id,
                        check_cancelled=check_cancelled,
                    ):
                        has_chunks = True

                        # Check for cancellation
                        if redis_manager.check_cancellation(conversation_id, run_id):
                            logger.info("Regenerate execution cancelled")
                            # Do not flush here - stop_generation saves from Redis snapshot to avoid duplicates
                            redis_manager.publish_event(
                                conversation_id,
                                run_id,
                                "end",
                                {
                                    "status": "cancelled",
                                    "message": "Regeneration cancelled by user",
                                },
                            )
                            return False  # Indicate cancellation

                        # Publish chunk event (run in executor to avoid blocking the event loop:
                        # sync Redis would block consumption of the model stream and can trigger
                        # "Model stream idle timeout" when the agent is streaming long code.)
                        serialized_tool_calls = []
                        if chunk.tool_calls:
                            for tool_call in chunk.tool_calls:
                                if hasattr(tool_call, "model_dump"):
                                    serialized_tool_calls.append(tool_call.model_dump())
                                elif hasattr(tool_call, "dict"):
                                    serialized_tool_calls.append(tool_call.dict())
                                else:
                                    serialized_tool_calls.append(tool_call)

                        payload = {
                            "content": chunk.message or "",
                            "citations_json": chunk.citations or [],
                            "tool_calls_json": serialized_tool_calls,
                            "thinking": chunk.thinking,
                        }

                        def _publish_chunk():
                            redis_manager.publish_event(
                                conversation_id, run_id, "chunk", payload
                            )

                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, _publish_chunk)

                    # Log completion of regeneration
                    if has_chunks:
                        logger.info("Regeneration completed successfully")
                    else:
                        logger.warning("No chunks received during regeneration")

                    return True  # Indicate successful completion
                except GenerationCancelled:
                    logger.info("Regenerate execution cancelled (GenerationCancelled)")
                    # Do not flush here - stop_generation saves from Redis snapshot to avoid duplicates
                    redis_manager.publish_event(
                        conversation_id,
                        run_id,
                        "end",
                        {
                            "status": "cancelled",
                            "message": "Regeneration cancelled by user",
                        },
                    )
                    return False  # Indicate cancellation

        # Run the async regeneration in a fresh event loop (asyncio.run)
        completed = self.run_async(run_regeneration())

        # Only publish completion event if not cancelled
        if completed:
            # Publish completion event
            redis_manager.publish_event(
                conversation_id,
                run_id,
                "end",
                {"status": "completed", "message": "Regeneration completed"},
            )

            # Set task status to completed
            redis_manager.set_task_status(conversation_id, run_id, "completed")

            logger.info("Background regenerate execution completed")
        else:
            logger.info("Background regenerate execution cancelled")

        # Return the completion status so on_success can check if it was cancelled
        return completed

    except Exception as e:
        logger.exception(
            "Background regenerate execution failed",
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
        )

        # Set task status to error
        try:
            redis_manager.set_task_status(conversation_id, run_id, "error")
        except Exception:
            logger.exception(
                "Failed to set task status to error",
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
            )

        # Ensure end event is always published
        try:
            redis_manager.publish_event(
                conversation_id,
                run_id,
                "end",
                {"status": "error", "message": str(e)},
            )
        except Exception:
            logger.exception(
                "Failed to publish error event to Redis",
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
            )
        raise
