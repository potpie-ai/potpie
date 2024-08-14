import asyncio
import json
from uuid6 import uuid7
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.agent_registry import get_agent

from app.modules.conversations.conversation.conversation_model import Conversation, ConversationStatus
from app.modules.conversations.message.message_model import Message, MessageType

from app.modules.conversations.conversation.conversation_schema import ConversationInfoResponse, ConversationResponse, CreateConversationRequest
from app.modules.conversations.message.message_schema import MessageRequest, MessageResponse

from app.modules.conversations.message.message_service import MessageService
from app.modules.intelligence.agents.agents_model import Agent
from app.modules.projects.projects_service import ProjectService


class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.project_service = ProjectService(db)
        self.message_service = MessageService(db)
    

    async def create_conversation(self, conversation: CreateConversationRequest):
        project_name = self.project_service.get_project_name(conversation.project_ids)
            
        # Create conversation
        conversation_id = str(uuid7())
        
        new_conversation = Conversation(
            id=conversation_id,
            user_id=conversation.user_id,
            title=project_name,
            status=ConversationStatus.ACTIVE,
            project_ids=conversation.project_ids,
            created_at=func.now(),
            updated_at=func.now(),
        )

        # Attach agents to the conversation
        agents = self.db.query(Agent).filter(Agent.id.in_(conversation.agent_ids)).all()
        
        new_conversation.agents.extend(agents)

        self.db.add(new_conversation)
        self.db.commit()

        # Create the initial message about the project creation
        initial_message_content = (
            f"Project {project_name} has been parsed successfully."
        )

        # Store the initial message
        self.message_service.create_message(
            conversation_id=conversation_id,
            content=initial_message_content,
            sender_id=None,
            message_type=MessageType.SYSTEM_GENERATED
        )

        # Perform a search using the DuckDuckGo agent
        agent = get_agent("langchain_duckduckgo")
        search_result = agent.run(f"Search for {project_name}")

        # Create the AI-generated message based on the search result
        self.message_service.create_message(
            conversation_id=conversation_id,
            content=search_result,
            sender_id=None,
            message_type=MessageType.AI_GENERATED
        )

        # Commit all messages in one transaction
        self.db.commit()

        # Return the generated ID and search result as the first message
        return conversation_id, search_result
    

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

    async def get_conversation(self, conversation_id: str) -> ConversationResponse:
        conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise ValueError("Conversation not found")

        return ConversationResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            status=conversation.status.value,
            project_ids=conversation.project_ids,
            agent_ids=conversation.agent_ids,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=[message.to_response() for message in conversation.messages]
        )

    async def get_conversation_info(self, conversation_id: str) -> ConversationInfoResponse:
        conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise ValueError("Conversation not found")

        return ConversationInfoResponse(
            id=conversation.id,
            agent_ids=conversation.agent_ids,
            project_ids=conversation.project_ids,
            total_messages=len(conversation.messages)
        )

    async def get_conversation_messages(self, conversation_id: str, start: int, limit: int) -> list[MessageResponse]:
        messages = (
            self.db.query(Message)
            .filter_by(conversation_id=conversation_id)
            .offset(start)
            .limit(limit)
            .all()
        )
        if not messages:
            return []

        return [message.to_response() for message in messages]

    async def delete_conversation(self, conversation_id: str) -> dict:
        conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise ValueError("Conversation not found")

        self.db.delete(conversation)
        self.db.commit()
        return {"status": "success"}
    

    async def regenerate_last_message(self, conversation_id: str):
        # Implement logic to regenerate the last message
        pass

    async def stop_generation(self, conversation_id: str):
        # Implement logic to stop message generation
        pass
