import logging
import os
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
)
from app.modules.intelligence.agents.debugging_agent import DebuggingAgent
from app.modules.intelligence.agents.intelligent_tool_using_orchestrator import (
    IntelligentToolUsingOrchestrator,
)
from app.modules.intelligence.agents.qna_agent import QNAAgent
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.duckduckgo_search_tool import DuckDuckGoTool
from app.modules.intelligence.tools.google_trends_tool import GoogleTrendsTool
from app.modules.intelligence.tools.wikipedia_tool import WikipediaTool
from app.modules.projects.projects_service import ProjectService

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
        project_service: ProjectService,
        history_manager: ChatHistoryService,
        orchestrator: IntelligentToolUsingOrchestrator,
        debugging_agent: DebuggingAgent,
        codebase_qna_agent: QNAAgent,
    ):
        self.db = db
        self.project_service = project_service
        self.history_manager = history_manager
        self.agents = {
            "chat_llm_orchestrator": orchestrator,
            "debugging_agent": debugging_agent,
            "codebase_qna_agent": codebase_qna_agent,
        }

    @classmethod
    def create(cls, db: Session):
        project_service = ProjectService(db)
        history_manager = ChatHistoryService(db)
        openai_key = cls._get_openai_key()
        orchestrator = cls._initialize_orchestrator(openai_key, db)
        debugging_agent = cls._initialize_debugging_agent(openai_key, db)
        qna_agent = cls._initialize_qna_agent(openai_key, db)
        return cls(
            db,
            project_service,
            history_manager,
            orchestrator,
            debugging_agent,
            qna_agent,
        )

    @staticmethod
    def _initialize_debugging_agent(openai_key: str, db: Session) -> DebuggingAgent:
        return DebuggingAgent(openai_key, db)

    @staticmethod
    def _initialize_qna_agent(openai_key: str, db: Session) -> QNAAgent:
        return QNAAgent(openai_key, db)

    @staticmethod
    def _get_openai_key() -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ConversationServiceError(
                "The OpenAI API key is not set in the environment variable 'OPENAI_API_KEY'."
            )
        return key

    @staticmethod
    def _initialize_orchestrator(
        openai_key: str, db: Session
    ) -> IntelligentToolUsingOrchestrator:
        tools = [GoogleTrendsTool(), WikipediaTool(), DuckDuckGoTool()]
        return IntelligentToolUsingOrchestrator(openai_key, tools, db)

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

            title = conversation.title.strip() if conversation.title else project_name

            conversation_id = self._create_conversation_record(
                conversation, title, user_id
            )

            await self._add_system_message(conversation_id, title, user_id)

            return conversation_id, "Conversation created successfully."
        except IntegrityError as e:
            logger.error(f"IntegrityError in create_conversation: {e}", exc_info=True)
            self.db.rollback()
            raise ConversationServiceError(
                "Failed to create conversation due to a database integrity error."
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in create_conversation: {e}", exc_info=True)
            self.db.rollback()
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
        self.db.add(new_conversation)
        self.db.commit()
        logger.info(
            f"Created new conversation with ID: {conversation_id}, title: {title}, user_id: {user_id}, agent_id: {conversation.agent_ids[0]}"
        )
        return conversation_id

    async def _add_system_message(
        self, conversation_id: str, project_name: str, user_id: str
    ):
        content = f"Project {project_name} has been parsed successfully."
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
            if message_type == MessageType.HUMAN:
                async for chunk in self._generate_and_stream_ai_response(
                    message.content, conversation_id, user_id
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
        self, conversation_id: str, user_id: str
    ) -> AsyncGenerator[str, None]:
        try:
            last_human_message = await self._get_last_human_message(conversation_id)
            if not last_human_message:
                raise MessageNotFoundError("No human message found to regenerate from")

            await self._archive_subsequent_messages(
                conversation_id, last_human_message.created_at
            )

            async for chunk in self._generate_and_stream_ai_response(
                last_human_message.content, conversation_id, user_id
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
            self.db.query(Message)
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
            self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.created_at > timestamp,
            ).update(
                {Message.status: MessageStatus.ARCHIVED}, synchronize_session="fetch"
            )
            self.db.commit()
            logger.info(
                f"Archived subsequent messages in conversation {conversation_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to archive messages in conversation {conversation_id}: {e}",
                exc_info=True,
            )
            self.db.rollback()
            raise ConversationServiceError(
                "Failed to archive subsequent messages."
            ) from e

    async def _generate_ai_response(
        self, query: str, conversation_id: str, user_id: str
    ) -> str:
        full_content = ""
        try:
            async for chunk in self.orchestrator.run(query, user_id, conversation_id):
                if chunk:
                    full_content += chunk
            logger.info(
                f"Generated AI response for conversation {conversation_id} for user {user_id}"
            )
            return full_content.strip()
        except Exception as e:
            logger.error(
                f"Failed to generate AI response for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError("Failed to generate AI response.") from e

    async def _generate_and_stream_ai_response(
        self, query: str, conversation_id: str, user_id: str
    ) -> AsyncGenerator[str, None]:
        conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
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
            async for chunk in agent.run(
                query, conversation.project_ids[0], user_id, conversation.id
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
            # Start a new transaction
            with self.db.begin():
                # Delete related messages first
                deleted_messages = (
                    self.db.query(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .delete()
                )

                # Delete the conversation
                deleted_conversation = (
                    self.db.query(Conversation)
                    .filter(Conversation.id == conversation_id)
                    .delete()
                )

                if deleted_conversation == 0:
                    raise ConversationNotFoundError(
                        f"Conversation with id {conversation_id} not found"
                    )

                # The transaction will be automatically committed if we reach this point

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
            raise

        except SQLAlchemyError as e:
            logger.error(f"Database error in delete_conversation: {e}", exc_info=True)
            # The transaction will be automatically rolled back
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to a database error"
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error in delete_conversation: {e}", exc_info=True)
            # Ensure rollback in case of any other exception
            self.db.rollback()
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to an unexpected error"
            ) from e

    async def get_conversation_info(
        self, conversation_id: str, user_id: str
    ) -> ConversationInfoResponse:
        try:
            conversation = (
                self.db.query(Conversation).filter_by(id=conversation_id).first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )
            total_messages = (
                self.db.query(Message)
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
                self.db.query(Conversation).filter_by(id=conversation_id).first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            messages = (
                self.db.query(Message)
                .filter_by(conversation_id=conversation_id)
                .filter_by(status=MessageStatus.ACTIVE)  # Only fetch active messages
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
                    status=message.status,  # Include the status field
                    created_at=message.created_at,
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
        # Implement the logic to stop the generation process
        # This might involve setting a flag in the orchestrator or cancelling an ongoing task
        logger.info(f"Attempting to stop generation for conversation {conversation_id}")
        return {"status": "success", "message": "Generation stop request received"}


    async def rename_conversation(self, conversation_id: str, new_title: str, user_id: str) -> dict:
        try:
            conversation = self.db.query(Conversation).filter_by(id=conversation_id, user_id=user_id).first()
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with id {conversation_id} not found")

            conversation.title = new_title
            conversation.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            logger.info(f"Renamed conversation {conversation_id} to '{new_title}' by user {user_id}")
            return {"status": "success", "message": f"Conversation renamed to '{new_title}'"}

        except SQLAlchemyError as e:
            logger.error(f"Database error in rename_conversation: {e}", exc_info=True)
            self.db.rollback()
            raise ConversationServiceError("Failed to rename conversation due to a database error") from e

        except Exception as e:
            logger.error(f"Unexpected error in rename_conversation: {e}", exc_info=True)
            self.db.rollback()
            raise ConversationServiceError("Failed to rename conversation due to an unexpected error") from e