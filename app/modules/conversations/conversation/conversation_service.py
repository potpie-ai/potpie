import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, List

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    ConversationStatus,
)
from app.modules.conversations.conversation.conversation_schema import (
    ConversationInfoResponse,
    CreateConversationRequest,
)
from app.modules.conversations.message.message_model import (
    Message,
    MessageStatus,
    MessageType,
)
from app.modules.conversations.message.message_schema import (
    MessageRequest,
    MessageResponse,
    NodeContext,
)
from app.modules.intelligence.agents.chat_agents.code_changes_agent import (
    CodeChangesAgent,
)
from app.modules.intelligence.agents.chat_agents.debugging_agent import DebuggingAgent
from app.modules.intelligence.agents.chat_agents.integration_test_agent import (
    IntegrationTestAgent,
)
from app.modules.intelligence.agents.chat_agents.qna_agent import QNAAgent
from app.modules.intelligence.agents.chat_agents.unit_test_agent import UnitTestAgent
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.posthog_helper import PostHogClient

logger = logging.getLogger(__name__)


class ConversationServiceError(Exception):
    """Base exception class for ConversationService errors."""


class ConversationNotFoundError(ConversationServiceError):
    """Raised when a conversation is not found."""


class MessageNotFoundError(ConversationServiceError):
    """Raised when a message is not found."""


class ConversationService:
    def __init__(
        self,
        db: Session,
        user_id: str,
        project_service: ProjectService,
        history_manager: ChatHistoryService,
        provider_service: ProviderService,
    ):
        self.sql_db = db
        self.user_id = user_id
        self.project_service = project_service
        self.history_manager = history_manager
        self.provider_service = provider_service
        self.agents = self._initialize_agents()

    @classmethod
    def create(cls, db: Session, user_id: str):
        project_service = ProjectService(db)
        history_manager = ChatHistoryService(db)
        provider_service = ProviderService(db, user_id)
        return cls(db, user_id, project_service, history_manager, provider_service)

    def _initialize_agents(self):
        mini_llm = self.provider_service.get_small_llm()
        reasoning_llm = self.provider_service.get_large_llm()
        return {
            "debugging_agent": DebuggingAgent(mini_llm, reasoning_llm, self.sql_db),
            "codebase_qna_agent": QNAAgent(mini_llm, reasoning_llm, self.sql_db),
            "unit_test_agent": UnitTestAgent(mini_llm, reasoning_llm, self.sql_db),
            "integration_test_agent": IntegrationTestAgent(
                mini_llm, reasoning_llm, self.sql_db
            ),
            "code_changes_agent": CodeChangesAgent(
                mini_llm, reasoning_llm, self.sql_db
            ),
        }

    async def create_conversation(
        self, conversation: CreateConversationRequest, user_id: str
    ) -> tuple[str, str]:
        try:
            if conversation.agent_ids[0] not in self.agents:
                raise ConversationServiceError(
                    f"Invalid agent_id: {conversation.agent_ids[0]}"
                )

            project_name = await self.project_service.get_project_name(
                conversation.project_ids
            )

            title = (
                conversation.title.strip().replace("Untitled", project_name)
                if conversation.title
                else project_name
            )

            conversation_id = self._create_conversation_record(
                conversation, title, user_id
            )

            await self._add_system_message(conversation_id, project_name, user_id)

            return conversation_id, "Conversation created successfully."
        except IntegrityError as e:
            logger.error(f"IntegrityError in create_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to create conversation due to a database integrity error."
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in create_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "An unexpected error occurred while creating the conversation."
            ) from e

    def _create_conversation_record(
        self, conversation: CreateConversationRequest, title: str, user_id: str
    ) -> str:
        conversation_id = str(uuid7())
        new_conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title,
            status=ConversationStatus.ACTIVE,
            project_ids=conversation.project_ids,
            agent_ids=conversation.agent_ids,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.sql_db.add(new_conversation)
        self.sql_db.commit()
        logger.info(
            f"Project id : {conversation.project_ids[0]} Created new conversation with ID: {conversation_id}, title: {title}, user_id: {user_id}, agent_id: {conversation.agent_ids[0]}"
        )
        provider_name = self.provider_service.get_llm_provider_name()
        PostHogClient().send_event(
            user_id,
            "create Conversation Event",
            {
                "project_ids": conversation.project_ids,
                "agent_ids": conversation.agent_ids,
                "llm": provider_name,
            },
        )
        return conversation_id

    async def _add_system_message(
        self, conversation_id: str, project_name: str, user_id: str
    ):
        content = f"You can now ask questions about the {project_name} repository."
        try:
            self.history_manager.add_message_chunk(
                conversation_id, content, MessageType.SYSTEM_GENERATED, user_id
            )
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.SYSTEM_GENERATED, user_id
            )
            logger.info(
                f"Added system message to conversation {conversation_id} for user {user_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to add system message to conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError(
                "Failed to add system message to the conversation."
            ) from e

    async def store_message(
        self,
        conversation_id: str,
        message: MessageRequest,
        message_type: MessageType,
        user_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            self.history_manager.add_message_chunk(
                conversation_id, message.content, message_type, user_id
            )
            self.history_manager.flush_message_buffer(
                conversation_id, message_type, user_id
            )
            logger.info(f"Stored message in conversation {conversation_id}")
            provider_name = self.provider_service.get_llm_provider_name()

            PostHogClient().send_event(
                user_id,
                "message post event",
                {"conversation_id": conversation_id, "llm": provider_name},
            )
            if message_type == MessageType.HUMAN:
                conversation = (
                    self.sql_db.query(Conversation)
                    .filter_by(id=conversation_id)
                    .first()
                )
                if not conversation:
                    raise ConversationNotFoundError(
                        f"Conversation with id {conversation_id} not found"
                    )

                repo_id = (
                    conversation.project_ids[0] if conversation.project_ids else None
                )
                if not repo_id:
                    raise ConversationServiceError(
                        "No project associated with this conversation"
                    )

                agent = self.agents.get(conversation.agent_ids[0])
                if not agent:
                    raise ConversationServiceError(
                        f"Invalid agent_id: {conversation.agent_ids[0]}"
                    )

                logger.info(
                    f"Running agent for repo_id: {repo_id} conversation_id: {conversation_id}"
                )
                async for chunk in agent.run(
                    message.content, repo_id, user_id, conversation.id, message.node_ids
                ):
                    yield chunk
        except Exception as e:
            logger.error(
                f"Error in store_message for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError(
                "Failed to store message or generate AI response."
            ) from e

    async def regenerate_last_message(
        self, conversation_id: str, user_id: str, node_ids: List[NodeContext] = []
    ) -> AsyncGenerator[str, None]:
        try:
            last_human_message = await self._get_last_human_message(conversation_id)
            if not last_human_message:
                raise MessageNotFoundError("No human message found to regenerate from")

            await self._archive_subsequent_messages(
                conversation_id, last_human_message.created_at
            )
            PostHogClient().send_event(
                user_id,
                "regenerate_conversation_event",
                {"conversation_id": conversation_id},
            )

            async for chunk in self._generate_and_stream_ai_response(
                last_human_message.content, conversation_id, user_id, node_ids
            ):
                yield chunk
        except MessageNotFoundError as e:
            logger.warning(
                f"No message to regenerate in conversation {conversation_id}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Error in regenerate_last_message for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError("Failed to regenerate last message.") from e

    async def _get_last_human_message(self, conversation_id: str):
        message = (
            self.sql_db.query(Message)
            .filter_by(conversation_id=conversation_id, type=MessageType.HUMAN)
            .order_by(Message.created_at.desc())
            .first()
        )
        if not message:
            logger.warning(f"No human message found in conversation {conversation_id}")
        return message

    async def _archive_subsequent_messages(
        self, conversation_id: str, timestamp: datetime
    ):
        try:
            self.sql_db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.created_at > timestamp,
            ).update(
                {Message.status: MessageStatus.ARCHIVED}, synchronize_session="fetch"
            )
            self.sql_db.commit()
            logger.info(
                f"Archived subsequent messages in conversation {conversation_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to archive messages in conversation {conversation_id}: {e}",
                exc_info=True,
            )
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to archive subsequent messages."
            ) from e

    async def _generate_and_stream_ai_response(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        conversation = (
            self.sql_db.query(Conversation).filter_by(id=conversation_id).first()
        )
        if not conversation:
            raise ConversationNotFoundError(
                f"Conversation with id {conversation_id} not found"
            )
        agent = self.agents.get(conversation.agent_ids[0])
        if not agent:
            raise ConversationServiceError(
                f"Invalid agent_id: {conversation.agent_ids[0]}"
            )

        try:
            logger.info(
                f"conversation_id: {conversation_id}Running agent {conversation.agent_ids[0]} with query: {query} "
            )
            async for chunk in agent.run(
                query, conversation.project_ids[0], user_id, conversation.id, node_ids
            ):
                if chunk:
                    yield chunk
            logger.info(
                f"Generated and streamed AI response for conversation {conversation.id} for user {user_id} using agent {conversation.agent_ids[0]}"
            )
        except Exception as e:
            logger.error(
                f"Failed to generate and stream AI response for conversation {conversation.id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError(
                "Failed to generate and stream AI response."
            ) from e

    async def delete_conversation(self, conversation_id: str, user_id: str) -> dict:
        try:
            # Use a nested transaction if one is already in progress
            with self.sql_db.begin_nested():
                # Delete related messages first
                deleted_messages = (
                    self.sql_db.query(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .delete(synchronize_session="fetch")
                )

                deleted_conversation = (
                    self.sql_db.query(Conversation)
                    .filter(Conversation.id == conversation_id)
                    .delete(synchronize_session="fetch")
                )

                if deleted_conversation == 0:
                    raise ConversationNotFoundError(
                        f"Conversation with id {conversation_id} not found"
                    )

            # If we get here, commit the transaction
            self.sql_db.commit()

            PostHogClient().send_event(
                user_id,
                "delete_conversation_event",
                {"conversation_id": conversation_id},
            )

            logger.info(
                f"Deleted conversation {conversation_id} and {deleted_messages} related messages"
            )
            return {
                "status": "success",
                "message": f"Conversation {conversation_id} and its messages have been permanently deleted.",
                "deleted_messages_count": deleted_messages,
            }

        except ConversationNotFoundError as e:
            logger.warning(str(e))
            self.sql_db.rollback()
            raise

        except SQLAlchemyError as e:
            logger.error(f"Database error in delete_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to a database error"
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error in delete_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to an unexpected error"
            ) from e

    async def get_conversation_info(
        self, conversation_id: str, user_id: str
    ) -> ConversationInfoResponse:
        try:
            conversation = (
                self.sql_db.query(Conversation).filter_by(id=conversation_id).first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )
            total_messages = (
                self.sql_db.query(Message)
                .filter_by(conversation_id=conversation_id, status=MessageStatus.ACTIVE)
                .count()
            )
            return ConversationInfoResponse(
                id=conversation.id,
                title=conversation.title,
                status=conversation.status,
                project_ids=conversation.project_ids,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                total_messages=total_messages,
                agent_ids=conversation.agent_ids,
            )
        except ConversationNotFoundError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(f"Error in get_conversation_info: {e}", exc_info=True)
            raise ConversationServiceError(
                f"Failed to get conversation info for {conversation_id}"
            ) from e

    async def get_conversation_messages(
        self, conversation_id: str, start: int, limit: int, user_id: str
    ) -> List[MessageResponse]:
        try:
            conversation = (
                self.sql_db.query(Conversation).filter_by(id=conversation_id).first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            messages = (
                self.sql_db.query(Message)
                .filter_by(conversation_id=conversation_id)
                .filter_by(status=MessageStatus.ACTIVE)
                .order_by(Message.created_at)
                .offset(start)
                .limit(limit)
                .all()
            )

            return [
                MessageResponse(
                    id=message.id,
                    conversation_id=message.conversation_id,
                    content=message.content,
                    sender_id=message.sender_id,
                    type=message.type,
                    status=message.status,
                    created_at=message.created_at,
                    citations=(
                        message.citations.split(",") if message.citations else None
                    ),
                )
                for message in messages
            ]
        except ConversationNotFoundError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(f"Error in get_conversation_messages: {e}", exc_info=True)
            raise ConversationServiceError(
                f"Failed to get messages for conversation {conversation_id}"
            ) from e

    async def stop_generation(self, conversation_id: str, user_id: str) -> dict:
        logger.info(f"Attempting to stop generation for conversation {conversation_id}")
        return {"status": "success", "message": "Generation stop request received"}

    async def rename_conversation(
        self, conversation_id: str, new_title: str, user_id: str
    ) -> dict:
        try:
            conversation = (
                self.sql_db.query(Conversation)
                .filter_by(id=conversation_id, user_id=user_id)
                .first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            conversation.title = new_title
            conversation.updated_at = datetime.now(timezone.utc)
            self.sql_db.commit()

            logger.info(
                f"Renamed conversation {conversation_id} to '{new_title}' by user {user_id}"
            )
            return {
                "status": "success",
                "message": f"Conversation renamed to '{new_title}'",
            }

        except SQLAlchemyError as e:
            logger.error(f"Database error in rename_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to rename conversation due to a database error"
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error in rename_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to rename conversation due to an unexpected error"
            ) from e
