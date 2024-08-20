import os
import logging
from typing import AsyncGenerator, Optional, List
from uuid6 import uuid7
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.modules.intelligence.agents.intelligent_tool_using_orchestrator import IntelligentToolUsingOrchestrator
from app.modules.intelligence.tools.duckduckgo_search_tool import DuckDuckGoTool
from app.modules.intelligence.tools.google_trends_tool import GoogleTrendsTool
from app.modules.intelligence.tools.wikipedia_tool import WikipediaTool
from app.modules.projects.projects_service import ProjectService
from app.modules.conversations.conversation.conversation_model import Conversation, ConversationStatus
from app.modules.conversations.message.message_model import Message, MessageType, MessageStatus
from app.modules.conversations.conversation.conversation_schema import CreateConversationRequest
from app.modules.conversations.message.message_schema import MessageRequest
from app.modules.intelligence.memory.postgres_history_manager import PostgresChatHistoryManager
from app.modules.intelligence.tools.tool_factory import create_tools

logger = logging.getLogger(__name__)

class ConversationServiceError(Exception):
    """Base exception class for ConversationService errors."""

class ConversationNotFoundError(ConversationServiceError):
    """Raised when a conversation is not found."""

class MessageNotFoundError(ConversationServiceError):
    """Raised when a message is not found."""

class ConversationService:
    def __init__(self, 
                 db: Session, 
                 project_service: ProjectService, 
                 history_manager: PostgresChatHistoryManager, 
                 orchestrator: IntelligentToolUsingOrchestrator):
        self.db = db
        self.project_service = project_service
        self.history_manager = history_manager
        self.orchestrator = orchestrator

    @classmethod
    def create(cls, db: Session):
        project_service = ProjectService(db)
        history_manager = PostgresChatHistoryManager(db)
        openai_key = cls._get_openai_key()
        orchestrator = cls._initialize_orchestrator(openai_key, db)
        return cls(db, project_service, history_manager, orchestrator)

    @staticmethod
    def _get_openai_key() -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ConversationServiceError("The OpenAI API key is not set in the environment variable 'OPENAI_API_KEY'.")
        return key

    @staticmethod
    def _initialize_orchestrator(openai_key: str, db: Session) -> IntelligentToolUsingOrchestrator:
        tools = [GoogleTrendsTool(), WikipediaTool(), DuckDuckGoTool()]
        return IntelligentToolUsingOrchestrator(openai_key, tools, db)


    async def create_conversation(self, conversation: CreateConversationRequest) -> tuple[str, str]:
        try:
            conversation_id = self._create_conversation_record(conversation)
            project_name = await self.project_service.get_project_name(conversation.project_ids)
            
            await self._add_system_message(conversation_id, project_name)
            await self._generate_initial_ai_response(conversation_id, project_name)

            return conversation_id, "Conversation created successfully."
        except IntegrityError as e:
            logger.error(f"IntegrityError in create_conversation: {e}", exc_info=True)
            self.db.rollback()
            raise ConversationServiceError("Failed to create conversation due to a database integrity error.") from e
        except Exception as e:
            logger.error(f"Unexpected error in create_conversation: {e}", exc_info=True)
            self.db.rollback()
            raise ConversationServiceError("An unexpected error occurred while creating the conversation.") from e

    def _create_conversation_record(self, conversation: CreateConversationRequest) -> str:
        conversation_id = str(uuid7())
        new_conversation = Conversation(
            id=conversation_id,
            user_id=conversation.user_id,
            title="",  # Title will be set later
            status=ConversationStatus.ACTIVE,
            project_ids=conversation.project_ids,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.db.add(new_conversation)
        self.db.commit()
        logger.info(f"Created new conversation with ID: {conversation_id}")
        return conversation_id

    async def _add_system_message(self, conversation_id: str, project_name: str):
        content = f"Project {project_name} has been parsed successfully."
        try:
            self.history_manager.add_message_chunk(conversation_id, content, MessageType.SYSTEM_GENERATED)
            self.history_manager.flush_message_buffer(conversation_id, MessageType.SYSTEM_GENERATED)
            logger.info(f"Added system message to conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to add system message to conversation {conversation_id}: {e}", exc_info=True)
            raise ConversationServiceError("Failed to add system message to the conversation.") from e

    async def _generate_initial_ai_response(self, conversation_id: str, project_name: str):
        query = f"Summarize the project: {project_name}"
        try:
            full_content = await self._generate_ai_response(query, conversation_id)
            self.history_manager.add_message_chunk(conversation_id, full_content, MessageType.AI_GENERATED)
            self.history_manager.flush_message_buffer(conversation_id, MessageType.AI_GENERATED)
            logger.info(f"Generated initial AI response for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to generate initial AI response for conversation {conversation_id}: {e}", exc_info=True)
            raise ConversationServiceError("Failed to generate initial AI response.") from e

    async def store_message(self, conversation_id: str, message: MessageRequest, message_type: MessageType, user_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        try:
            self.history_manager.add_message_chunk(conversation_id, message.content, message_type, user_id)
            self.history_manager.flush_message_buffer(conversation_id, message_type, user_id)
            logger.info(f"Stored message in conversation {conversation_id}")

            if message_type == MessageType.HUMAN:
                async for chunk in self._generate_and_stream_ai_response(message.content, conversation_id):
                    yield chunk
        except Exception as e:
            logger.error(f"Error in store_message for conversation {conversation_id}: {e}", exc_info=True)
            raise ConversationServiceError("Failed to store message or generate AI response.") from e

    async def regenerate_last_message(self, conversation_id: str) -> AsyncGenerator[str, None]:
        try:
            last_human_message = await self._get_last_human_message(conversation_id)
            if not last_human_message:
                raise MessageNotFoundError("No human message found to regenerate from")

            await self._archive_subsequent_messages(conversation_id, last_human_message.created_at)
            
            async for chunk in self._generate_and_stream_ai_response(last_human_message.content, conversation_id):
                yield chunk
        except MessageNotFoundError as e:
            logger.warning(f"No message to regenerate in conversation {conversation_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in regenerate_last_message for conversation {conversation_id}: {e}", exc_info=True)
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

    async def _archive_subsequent_messages(self, conversation_id: str, timestamp: datetime):
        try:
            self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.created_at > timestamp
            ).update({Message.status: MessageStatus.ARCHIVED}, synchronize_session='fetch')
            self.db.commit()
            logger.info(f"Archived subsequent messages in conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to archive messages in conversation {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            raise ConversationServiceError("Failed to archive subsequent messages.") from e

    async def _generate_ai_response(self, query: str, conversation_id: str) -> str:
        full_content = ""
        try:
            async for chunk in self.orchestrator.run(query, "user_id", conversation_id):
                if chunk:
                    full_content += chunk
            logger.info(f"Generated AI response for conversation {conversation_id}")
            return full_content.strip()
        except Exception as e:
            logger.error(f"Failed to generate AI response for conversation {conversation_id}: {e}", exc_info=True)
            raise ConversationServiceError("Failed to generate AI response.") from e

    async def _generate_and_stream_ai_response(self, query: str, conversation_id: str) -> AsyncGenerator[str, None]:
        full_content = ""
        try:
            async for chunk in self.orchestrator.run(query, "user_id", conversation_id):
                if chunk:
                    full_content += chunk
                    yield chunk  # Stream to the client

            self.history_manager.add_message_chunk(conversation_id, full_content.strip(), MessageType.AI_GENERATED)
            self.history_manager.flush_message_buffer(conversation_id, MessageType.AI_GENERATED)
            logger.info(f"Generated and streamed AI response for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to generate and stream AI response for conversation {conversation_id}: {e}", exc_info=True)
            raise ConversationServiceError("Failed to generate and stream AI response.") from e