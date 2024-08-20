import os
import logging
import asyncio
from typing import AsyncGenerator, Optional, List
from uuid6 import uuid7
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.modules.intelligence.agents.intelligent_tool_using_orchestrator import IntelligentToolUsingOrchestrator
from app.modules.projects.projects_service import ProjectService
from app.modules.conversations.conversation.conversation_model import Conversation, ConversationStatus
from app.modules.conversations.message.message_model import Message, MessageType, MessageStatus
from app.modules.conversations.conversation.conversation_schema import CreateConversationRequest, ConversationInfoResponse
from app.modules.conversations.message.message_schema import MessageRequest, MessageResponse
from app.modules.conversations.message.message_service import MessageService
from app.modules.intelligence.tools.duckduckgo_search_tool import DuckDuckGoTool
from app.modules.intelligence.tools.google_trends_tool import GoogleTrendsTool
from app.modules.intelligence.tools.wikipedia_tool import WikipediaTool

# Set up logging
logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.project_service = ProjectService(db)
        self.message_service = MessageService(db)
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("The OpenAI API key is not set in the environment variable 'OPENAI_API_KEY'.")
        self.orchestrator = self._initialize_orchestrator()

    def _initialize_orchestrator(self) -> IntelligentToolUsingOrchestrator:
        tools = [
            GoogleTrendsTool(),
            WikipediaTool(),
            DuckDuckGoTool(),
        ]
        return IntelligentToolUsingOrchestrator(self.openai_key, tools, self.db)

    async def run_tool_using_orchestrator(self, query: str, user_id: str, conversation_id: str) -> AsyncGenerator[str, None]:
        async for chunk in self.orchestrator.run(query, user_id, conversation_id):
            yield chunk

    async def message_stream(self, conversation_id: str, query: str) -> AsyncGenerator[str, None]:
        try:
            full_content = ""
            async for content_update in self.run_tool_using_orchestrator(query, user_id="user_id", conversation_id=conversation_id):
                if content_update:
                    full_content += content_update
                    yield content_update
            await self.message_service.create_message(
                conversation_id,
                full_content.strip(),
                MessageType.AI_GENERATED
            )
        except Exception as e:
            logger.error(f"Error in message_stream: {e}")
            raise e

    async def create_conversation(self, conversation: CreateConversationRequest) -> tuple[str, str]:
        try:
            project_ids = conversation.project_ids
            project_name = await self.project_service.get_project_name(project_ids)
            conversation_id = str(uuid7())
            new_conversation = Conversation(
                id=conversation_id,
                user_id=conversation.user_id,
                title=project_name,
                status=ConversationStatus.ACTIVE,
                project_ids=project_ids,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            self.db.add(new_conversation)
            self.db.commit()
            await self.message_service.create_message(
                conversation_id,
                f"Project {project_name} has been parsed successfully.",
                MessageType.SYSTEM_GENERATED,
                sender_id=None,
            )
            # Await the async task to ensure it completes properly
            await self._async_create_conversation_post_commit(conversation_id, project_name)
            return conversation_id, "Conversation created successfully."
        except IntegrityError as e:
            logger.error(f"IntegrityError in create_conversation: {e}")
            self.db.rollback()
            raise e
        except Exception as e:
            logger.error(f"Error in create_conversation: {e}")
            self.db.rollback()
            raise e

    async def _async_create_conversation_post_commit(self, conversation_id: str, project_name: str):
        try:
            search_result = []
            async for result in self.run_tool_using_orchestrator(project_name, "user_id", conversation_id):
                if result:
                    search_result.append(result)
            combined_search_result = "\n".join(search_result)
            await self.message_service.create_message(conversation_id, combined_search_result, MessageType.AI_GENERATED)
        except Exception as e:
            logger.error(f"Error in async processing after transaction: {e}")

    async def store_message(self, conversation_id: str, message: MessageRequest, message_type: MessageType, user_id: Optional[str] = None) -> Message:
        try:
            return await self.message_service.create_message(conversation_id, message.content, message_type, user_id)
        except IntegrityError as e:
            logger.error(f"IntegrityError in store_message: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error in store_message: {e}")
            raise e

    async def regenerate_last_message(self, conversation_id: str) -> AsyncGenerator[str, None]:
        try:
            last_human_message = (
                self.db.query(Message)
                .filter_by(conversation_id=conversation_id, type=MessageType.HUMAN)
                .order_by(Message.created_at.desc())
                .first()
            )
            if not last_human_message:
                raise ValueError("No human message found to regenerate from")

            self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.created_at > last_human_message.created_at
            ).update({Message.status: MessageStatus.ARCHIVED}, synchronize_session='fetch')
            self.db.commit()

            full_content = ""
            async for chunk in self.run_tool_using_orchestrator(last_human_message.content, "user_id", conversation_id):
                if chunk:
                    full_content += chunk
                    yield f"data: {{'content': {chunk}}}\n\n"
            await self.message_service.create_message(conversation_id, full_content.strip(), MessageType.AI_GENERATED)
        except IntegrityError as e:
            logger.error(f"IntegrityError in regenerate_last_message: {e}")
            self.db.rollback()
            raise e
        except Exception as e:
            logger.error(f"Error in regenerate_last_message: {e}")
            self.db.rollback()
            raise e

    def get_conversation_info(self, conversation_id: str) -> ConversationInfoResponse:
        try:
            conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
            if not conversation:
                raise ValueError("Conversation not found")
            # Calculate total_messages based on active messages
            total_messages = self.db.query(Message).filter_by(conversation_id=conversation_id, status=MessageStatus.ACTIVE).count()
            return ConversationInfoResponse(
                id=conversation.id,
                title=conversation.title,
                status=conversation.status,
                project_ids=conversation.project_ids,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                total_messages=total_messages
            )
        except Exception as e:
            logger.error(f"Error in get_conversation_info: {e}")
            raise e

    def get_conversation_messages(self, conversation_id: str, start: int, limit: int) -> List[MessageResponse]:
        try:
            messages = (
                self.db.query(Message)
                .filter_by(conversation_id=conversation_id, status=MessageStatus.ACTIVE)
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
                    created_at=message.created_at
                ) for message in messages
            ]
        except Exception as e:
            logger.error(f"Error in get_conversation_messages: {e}")
            raise e

    def delete_conversation(self, conversation_id: str) -> dict:
        try:
            conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
            if not conversation:
                raise ValueError("Conversation not found")
            self.db.delete(conversation)
            self.db.commit()
            return {"status": "success"}
        except IntegrityError as e:
            logger.error(f"IntegrityError in delete_conversation: {e}")
            self.db.rollback()
            raise e
        except Exception as e:
            logger.error(f"Error in delete_conversation: {e}")
            self.db.rollback()
            raise e

    def stop_generation(self, conversation_id: str):
        pass