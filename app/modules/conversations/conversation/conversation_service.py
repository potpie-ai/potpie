import asyncio
import json
import uuid
from datetime import datetime
from fastapi import Depends
from sqlalchemy import func
from sqlalchemy.orm import Session


from app.core.base_service import BaseService
from app.core.database import get_db
from app.modules.conversations.conversation.conversation_model import Conversation, ConversationStatus
from app.modules.conversations.conversation.conversation_schema import CreateConversationRequest
from app.modules.conversations.message.message_model import Message
from app.modules.conversations.message.message_schema import MessageRequest
from app.modules.conversations.message.message_service import MessageService
from app.modules.intelligence.agents.agent_registry import get_agent
from app.modules.projects.projects_service import ProjectService


class ConversationService(BaseService):
    def __init__(self, db: Session):
        super().__init__(db)
        self.project_service = ProjectService(db)
        self.message_service = MessageService(db)

    async def create_conversation(self, conversation: CreateConversationRequest):
        try:
            project_name = self.project_service.get_project_name(conversation.project_ids)
            conversation_id = str(uuid.uuid4())
            new_conversation = Conversation(
                id=conversation_id,
                user_id=conversation.user_id,
                title=project_name,
                status=ConversationStatus.ACTIVE,
                project_ids=conversation.project_ids,
                agent_ids=conversation.agent_ids,
                created_at=func.now(),
                updated_at=func.now(),
            )
            self.db.add(new_conversation)
            self.db.commit()
            initial_message_content = (
                f"Project '{project_name}' has been created successfully. "
                f"Agents: {', '.join(conversation.agent_ids)}."
            )
            self.message_service.create_message(
                conversation_id=conversation_id,
                content=initial_message_content,
                sender_id=conversation.user_id,
                message_type="HUMAN"
            )
            agent = get_agent("langchain_duckduckgo")
            search_result = agent.run(f"Search for {project_name}")
            self.message_service.create_message(
                conversation_id=conversation_id,
                content=search_result,
                sender_id=None,
                message_type="AI_GENERATED"
            )
            self.db.commit()
            return conversation_id, search_result
        except Exception as e:
            self.db.rollback()
            raise e

    async def store_message(self, conversation_id: str, message: MessageRequest, user_id: str):
        new_message = Message(
            conversation_id=conversation_id,
            content=message.content,
            type="AI_GENERATED",
            created_at=datetime.now()
        )
        
        self.db.add(new_message)
        self.db.commit()
        self.db.refresh(new_message)
        
        return new_message

    async def generate_message_content(self):
        content_parts = ["string (part 1)", "string (part 2)", "string (part 3)"]
        for part in content_parts:
            await asyncio.sleep(1)
            yield f"data: {json.dumps({'content': part})}\n\n"

    async def message_stream(self, conversation_id: str):
        metadata = {
            "message_id": "mock-message-id",
            "conversation_id": conversation_id,
            "type": "AI_GENERATED",
            "reason": "STREAM"
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        async for content_update in self.generate_message_content():
            yield content_update
