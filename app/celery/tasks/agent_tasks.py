from typing import Optional, List

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.utils.logger import setup_logger, log_context

logger = setup_logger(__name__)


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

    with log_context(conversation_id=conversation_id, user_id=user_id, run_id=run_id):
        logger.info("Starting background agent execution")

        try:
            # ✅ Set running status INSIDE try
            redis_manager.set_task_status(conversation_id, run_id, "running")

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
                from app.modules.users.user_model import User

                async with self.async_db() as async_db:
                    user_service = UserService(self.db)
                    user = user_service.get_user_by_uid(user_id)

                    if not user:
                        user = (
                            self.db.query(User)
                            .filter(User.uid == user_id)
                            .first()
                        )
                        if not user:
                            logger.warning(
                                "User not found, using empty email",
                                user_id=user_id,
                            )

                    user_email = user.email if user and user.email else ""

                    conversation_store = ConversationStore(self.db, async_db)
                    message_store = MessageStore(self.db, async_db)

                    service = ConversationService.create(
                        conversation_store=conversation_store,
                        message_store=message_store,
                        db=self.db,
                        user_id=user_id,
                        user_email=user_email,
                    )

                    message_request = MessageRequest(
                        content=query,
                        node_ids=node_ids,
                        attachment_ids=attachment_ids or None,
                    )

                    redis_manager.publish_event(
                        conversation_id,
                        run_id,
                        "start",
                        {
                            "agent_id": agent_id or "default",
                            "status": "processing",
                            "message": "Starting agent execution",
                        },
                    )

                    async for chunk in service.store_message(
                        conversation_id,
                        message_request,
                        MessageType.HUMAN,
                        user_id,
                        stream=True,
                    ):
                        if redis_manager.check_cancellation(conversation_id, run_id):
                            redis_manager.publish_event(
                                conversation_id,
                                run_id,
                                "end",
                                {
                                    "status": "cancelled",
                                    "message": "Execution cancelled by user",
                                },
                            )
                            return False

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

                    return True

            completed = self.run_async(run_agent())

            if completed:
                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "end",
                    {
                        "status": "completed",
                        "message": "Agent execution completed",
                    },
                )
                redis_manager.set_task_status(conversation_id, run_id, "completed")
                logger.info("Background agent execution completed")
            else:
                logger.info("Background agent execution cancelled")

            return completed

        except Exception as e:
            logger.exception(
                "Background agent execution failed",
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
            )

            try:
                redis_manager.set_task_status(
                    conversation_id,
                    run_id,
                    "error",
                )
            except Exception:
                logger.exception(
                    "Failed to update task status to error",
                    conversation_id=conversation_id,
                    run_id=run_id,
                )

            try:
                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "end",
                    {"status": "error", "message": str(e)},
                )
            except Exception:
                logger.exception(
                    "Failed to publish error event",
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

    with log_context(conversation_id=conversation_id, user_id=user_id, run_id=run_id):
        logger.info("Starting background regenerate execution")

        try:
            # ✅ Set running status INSIDE try
            redis_manager.set_task_status(conversation_id, run_id, "running")

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

                async with self.async_db() as async_db:
                    user_service = UserService(self.db)
                    user = user_service.get_user_by_uid(user_id)
                    user_email = user.email if user and user.email else ""

                    conversation_store = ConversationStore(self.db, async_db)
                    message_store = MessageStore(self.db, async_db)

                    service = ConversationService.create(
                        conversation_store=conversation_store,
                        message_store=message_store,
                        db=self.db,
                        user_id=user_id,
                        user_email=user_email,
                    )

                    redis_manager.publish_event(
                        conversation_id,
                        run_id,
                        "start",
                        {
                            "agent_id": "regenerate",
                            "status": "processing",
                            "message": "Starting regeneration",
                        },
                    )

                    async for chunk in service.regenerate_last_message_background(
                        conversation_id, node_ids, attachment_ids
                    ):
                        if redis_manager.check_cancellation(conversation_id, run_id):
                            redis_manager.publish_event(
                                conversation_id,
                                run_id,
                                "end",
                                {
                                    "status": "cancelled",
                                    "message": "Regeneration cancelled by user",
                                },
                            )
                            return False

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

                    return True

            completed = self.run_async(run_regeneration())

            if completed:
                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "end",
                    {
                        "status": "completed",
                        "message": "Regeneration completed",
                    },
                )
                redis_manager.set_task_status(conversation_id, run_id, "completed")
                logger.info("Background regenerate execution completed")
            else:
                logger.info("Background regenerate execution cancelled")

            return completed

        except Exception as e:
            logger.exception(
                "Background regenerate execution failed",
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
            )

            try:
                redis_manager.set_task_status(
                    conversation_id,
                    run_id,
                    "error",
                )
            except Exception:
                logger.exception(
                    "Failed to update regenerate task status to error",
                    conversation_id=conversation_id,
                    run_id=run_id,
                )

            try:
                redis_manager.publish_event(
                    conversation_id,
                    run_id,
                    "end",
                    {"status": "error", "message": str(e)},
                )
            except Exception:
                logger.exception(
                    "Failed to publish regenerate error event",
                    conversation_id=conversation_id,
                    run_id=run_id,
                )

            raise
