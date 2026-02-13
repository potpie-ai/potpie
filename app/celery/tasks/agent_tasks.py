from typing import Optional, List

import logfire
from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.utils.logger import setup_logger, log_context
from app.modules.intelligence.tracing.logfire_tracer import logfire_trace_metadata
from app.modules.intelligence.provider.openrouter_usage_context import (
    init_usage_context,
    get_and_clear_usages,
    estimate_cost_for_log,
)

logger = setup_logger(__name__)


def _record_openrouter_cost_in_logfire(usages: List[dict], outcome: str) -> float:
    """
    Compute total OpenRouter cost from usages and record as Logfire span attribute.
    
    Args:
        usages: List of usage dicts (from get_and_clear_usages)
        outcome: "completed" | "cancelled" | "error"
    
    Returns:
        total_cost: Sum of API-returned costs (0.0 if none)
    """
    total_cost = 0.0
    for u in usages:
        c = u.get("cost")
        if c is not None:
            total_cost += float(c)
    
    # Record cost as Logfire span attribute so it appears in the trace
    with logfire.span(
        "agent_run_usage",
        actual_cost=total_cost,
        outcome=outcome,
        usage_count=len(usages),
    ):
        pass
    
    return total_cost


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

    # Look up project_id from conversation so every span in this task gets it
    project_id = None
    try:
        from app.modules.conversations.conversation.conversation_model import Conversation
        conv = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv and conv.project_ids:
            project_id = conv.project_ids[0]
    except Exception:
        logger.debug("Could not resolve project_id for conversation %s", conversation_id)

    # Logfire: set trace metadata so all Pydantic AI / LiteLLM spans are queryable by user_id, etc.
    with logfire_trace_metadata(
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        agent_id=agent_id or "default",
        project_id=project_id,
    ):
        # Set up logging context with domain IDs
        with log_context(conversation_id=conversation_id, user_id=user_id, run_id=run_id):
            logger.info("Starting background agent execution")

            # Set task status to indicate task has started
            redis_manager.set_task_status(conversation_id, run_id, "running")

            # Collect OpenRouter usage so we can send it in the end event (API will log it)
            init_usage_context()

            try:
                # Execute agent with Redis publishing
                async def run_agent():
                    from app.modules.conversations.conversation.conversation_service import (
                        ConversationService,
                    )
                    from app.modules.users.user_service import UserService
                    from app.modules.conversations.message.message_model import MessageType
                    from app.modules.conversations.message.message_schema import (
                        MessageRequest,
                    )
                    from app.modules.conversations.conversation.conversation_store import (
                        ConversationStore,
                    )
                    from app.modules.conversations.message.message_store import MessageStore

                    # Use BaseTask's context manager to get a fresh, non-pooled async session
                    # This avoids asyncpg Future binding issues across tasks sharing the same event loop
                    async with self.async_db() as async_db:
                        # Get user email for service creation
                        from app.modules.users.user_model import User

                        user_service = UserService(self.db)
                        user = user_service.get_user_by_uid(user_id)

                        # Debug: Direct query to verify user exists
                        if not user:
                            direct_user = (
                                self.db.query(User).filter(User.uid == user_id).first()
                            )
                            if direct_user:
                                logger.warning(
                                    f"UserService.get_user_by_uid returned None but direct query found user: {direct_user.uid}, email: {direct_user.email}"
                                )
                                user = direct_user
                            else:
                                logger.warning(
                                    f"User not found in database for user_id: {user_id}. Using empty string as fallback."
                                )

                        conversation_store = ConversationStore(self.db, async_db)
                        message_store = MessageStore(self.db, async_db)

                        # Handle missing user or email gracefully
                        if not user:
                            user_email = ""
                        elif not user.email:
                            logger.warning(
                                f"User found but email is None/empty for user_id: {user_id}, "
                                f"user.uid: {user.uid if user else 'N/A'}, "
                                f"email value: {repr(user.email) if user else 'N/A'}. "
                                f"Using empty string as fallback."
                            )
                            user_email = ""
                        else:
                            user_email = user.email
                            logger.debug(
                                f"Retrieved user email: {user_email} for user_id: {user_id}"
                            )

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
                                logger.info("Agent execution cancelled")

                                # Flush any buffered AI response chunks before cancelling
                                try:
                                    message_id = (
                                        service.history_manager.flush_message_buffer(
                                            conversation_id, MessageType.AI_GENERATED
                                        )
                                    )
                                    if message_id:
                                        logger.debug(
                                            "Flushed partial AI response for cancelled task",
                                            message_id=message_id,
                                        )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to flush message buffer on cancellation",
                                        error=str(e),
                                    )
                                # Continue with cancellation even if flush fails
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

                        return True  # Indicate successful completion

                # Run the async agent execution on the worker's long-lived loop
                completed = self.run_async(run_agent())

                # Collect OpenRouter usage and record cost in Logfire (for all outcomes)
                usages = get_and_clear_usages()
                total_cost = _record_openrouter_cost_in_logfire(
                    usages, "completed" if completed else "cancelled"
                )

                # Only publish completion event if not cancelled
                if completed:
                    # Include OpenRouter usage in end event so API (uvicorn) can log it
                    end_payload = {
                        "status": "completed",
                        "message": "Agent execution completed",
                    }
                    if usages:
                        end_payload["usage_json"] = usages
                        # Log per-usage details (total_cost already computed and recorded in Logfire)
                        for u in usages:
                            c = u.get("cost")
                            pt = u.get("prompt_tokens", 0) or 0
                            ct = u.get("completion_tokens", 0) or 0
                            if c is not None:
                                cost_str = f", cost={c} credits"
                            else:
                                est = estimate_cost_for_log(pt, ct) if (pt or ct) else 0.0
                                cost_str = f", cost≈{est} credits (estimated, not in run total)" if (pt or ct) else ""
                            msg = (
                                f"[OpenRouter usage] model={u.get('model', '')} "
                                f"prompt_tokens={pt} completion_tokens={ct} "
                                f"total_tokens={u.get('total_tokens', 0)}{cost_str}"
                            )
                            logger.info(msg)
                            print(msg, flush=True)
                        if usages:
                            summary = f"[LLM cost this run] total={total_cost} credits (see lines above for per-call breakdown)"
                            logger.info(summary)
                            print(summary, flush=True)
                    redis_manager.publish_event(
                        conversation_id,
                        run_id,
                        "end",
                        end_payload,
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

                # Collect OpenRouter usage and record partial cost in Logfire (even on error)
                try:
                    usages = get_and_clear_usages()
                    total_cost = _record_openrouter_cost_in_logfire(usages, "error")
                    
                    # Log partial cost to Celery logs so failed runs also show cost
                    if usages:
                        for u in usages:
                            c = u.get("cost")
                            pt = u.get("prompt_tokens", 0) or 0
                            ct = u.get("completion_tokens", 0) or 0
                            if c is not None:
                                cost_str = f", cost={c} credits"
                            else:
                                est = estimate_cost_for_log(pt, ct) if (pt or ct) else 0.0
                                cost_str = f", cost≈{est} credits (estimated)" if (pt or ct) else ""
                            msg = (
                                f"[OpenRouter usage - partial] model={u.get('model', '')} "
                                f"prompt_tokens={pt} completion_tokens={ct} "
                                f"total_tokens={u.get('total_tokens', 0)}{cost_str}"
                            )
                            logger.info(msg)
                            print(msg, flush=True)
                        logger.info(
                            f"[LLM cost - partial before error] total={total_cost} credits"
                        )
                        print(
                            f"[LLM cost - partial before error] total={total_cost} credits",
                            flush=True,
                        )
                except Exception as cost_error:
                    logger.warning(f"Failed to record partial cost on error: {cost_error}")

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
) -> None:
    """Execute regeneration in the background and publish results to Redis streams"""
    redis_manager = RedisStreamManager()

    # Look up project_id from conversation so every span in this task gets it
    project_id = None
    try:
        from app.modules.conversations.conversation.conversation_model import Conversation
        conv = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv and conv.project_ids:
            project_id = conv.project_ids[0]
    except Exception:
        logger.debug("Could not resolve project_id for conversation %s", conversation_id)

    # Logfire: set trace metadata so all Pydantic AI / LiteLLM spans are queryable by user_id, etc.
    with logfire_trace_metadata(
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        agent_id="regenerate",
        project_id=project_id,
    ):
        # Set up logging context with domain IDs
        with log_context(conversation_id=conversation_id, user_id=user_id, run_id=run_id):
            logger.info("Starting background regenerate execution")

            # Set task status to indicate task has started
            redis_manager.set_task_status(conversation_id, run_id, "running")

            # Collect OpenRouter usage so we can record cost in Logfire
            init_usage_context()

            try:
                # Execute regeneration with Redis publishing
                async def run_regeneration():
                    from app.modules.conversations.conversation.conversation_service import (
                        ConversationService,
                    )
                    from app.modules.users.user_service import UserService
                    from app.modules.conversations.conversation.conversation_store import (
                        ConversationStore,
                    )
                    from app.modules.conversations.message.message_store import MessageStore
                    from app.modules.conversations.message.message_model import MessageType

                    # Use BaseTask's context manager to get a fresh, non-pooled async session
                    # This avoids asyncpg Future binding issues across tasks sharing the same event loop
                    async with self.async_db() as async_db:
                        # Get user email for service creation
                        user_service = UserService(self.db)
                        user = user_service.get_user_by_uid(user_id)
                        # Handle missing user or email gracefully
                        if not user:
                            logger.warning(
                                f"User not found for user_id: {user_id}. Using empty string as fallback."
                            )
                            user_email = ""
                        elif not user.email:
                            logger.warning(
                                f"User found but email is None/empty for user_id: {user_id}, user object: {user}, email value: {repr(user.email)}. Using empty string as fallback."
                            )
                            user_email = ""
                        else:
                            user_email = user.email
                            logger.debug(
                                f"Retrieved user email: {user_email} for user_id: {user_id}"
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
                                logger.info("Regenerate execution cancelled")

                                # Flush any buffered AI response chunks before cancelling
                                try:
                                    message_id = service.history_manager.flush_message_buffer(
                                        conversation_id, MessageType.AI_GENERATED
                                    )
                                    if message_id:
                                        logger.debug(
                                            "Flushed partial AI response for cancelled regenerate",
                                            message_id=message_id,
                                        )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to flush message buffer on cancellation",
                                        error=str(e),
                                    )
                                # Continue with cancellation even if flush fails

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

                        # Log completion of regeneration
                        if has_chunks:
                            logger.info("Regeneration completed successfully")
                        else:
                            logger.warning("No chunks received during regeneration")

                        return True  # Indicate successful completion

                # Run the async regeneration on the worker's long-lived loop
                completed = self.run_async(run_regeneration())

                # Collect OpenRouter usage and record cost in Logfire (for all outcomes)
                usages = get_and_clear_usages()
                total_cost = _record_openrouter_cost_in_logfire(
                    usages, "completed" if completed else "cancelled"
                )

                # Log usage to Celery logs (for both completed and cancelled)
                if usages:
                    for u in usages:
                        c = u.get("cost")
                        pt = u.get("prompt_tokens", 0) or 0
                        ct = u.get("completion_tokens", 0) or 0
                        if c is not None:
                            cost_str = f", cost={c} credits"
                        else:
                            est = estimate_cost_for_log(pt, ct) if (pt or ct) else 0.0
                            cost_str = f", cost≈{est} credits (estimated)" if (pt or ct) else ""
                        msg = (
                            f"[OpenRouter usage] model={u.get('model', '')} "
                            f"prompt_tokens={pt} completion_tokens={ct} "
                            f"total_tokens={u.get('total_tokens', 0)}{cost_str}"
                        )
                        logger.info(msg)
                        print(msg, flush=True)
                    logger.info(
                        f"[LLM cost this run] total={total_cost} credits"
                    )
                    print(
                        f"[LLM cost this run] total={total_cost} credits",
                        flush=True,
                    )

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

                # Collect OpenRouter usage and record partial cost in Logfire (even on error)
                try:
                    usages = get_and_clear_usages()
                    total_cost = _record_openrouter_cost_in_logfire(usages, "error")
                    
                    # Log partial cost to Celery logs so failed runs also show cost
                    if usages:
                        for u in usages:
                            c = u.get("cost")
                            pt = u.get("prompt_tokens", 0) or 0
                            ct = u.get("completion_tokens", 0) or 0
                            if c is not None:
                                cost_str = f", cost={c} credits"
                            else:
                                est = estimate_cost_for_log(pt, ct) if (pt or ct) else 0.0
                                cost_str = f", cost≈{est} credits (estimated)" if (pt or ct) else ""
                            msg = (
                                f"[OpenRouter usage - partial] model={u.get('model', '')} "
                                f"prompt_tokens={pt} completion_tokens={ct} "
                                f"total_tokens={u.get('total_tokens', 0)}{cost_str}"
                            )
                            logger.info(msg)
                            print(msg, flush=True)
                        logger.info(
                            f"[LLM cost - partial before error] total={total_cost} credits"
                        )
                        print(
                            f"[LLM cost - partial before error] total={total_cost} credits",
                            flush=True,
                        )
                except Exception as cost_error:
                    logger.warning(f"Failed to record partial cost on error: {cost_error}")

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
