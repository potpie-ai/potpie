from typing import List
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.modules.conversations.conversation.service import ConversationService
from app.modules.conversations.message.schema import MessageRequest, MessageResponse
from .schema import CreateConversationRequest, CreateConversationResponse, ConversationResponse, ConversationInfoResponse

class ConversationController:

    def __init__(self):
        self.service = ConversationService()

    async def post_message(self, conversation_id: str, message: MessageRequest, db: Session, user_id: str):
        stored_message = await self.service.store_message(conversation_id, message, db, user_id)
        message_stream = self.service.message_stream(conversation_id)
        return StreamingResponse(message_stream, media_type="text/event-stream")
    
    
    async def create_conversation(self, conversation: CreateConversationRequest) -> CreateConversationResponse:
        conversation_id, message = await self.service.create_conversation(conversation)
        if isinstance(message, str):
            return CreateConversationResponse(message=message, conversation_id=conversation_id)
        else:
            raise ValueError("Message returned from service is not a string")

    async def get_conversation(self, conversation_id: str) -> ConversationResponse:
        return ConversationResponse(
            id=conversation_id,
            user_id="mock-user-id",
            title="Mock Conversation Title",
            status="active",
            project_ids=["project1", "project2"],
            agent_ids=["agent1", "agent2"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            messages=[]
        )

    async def get_conversation_info(self, conversation_id: str) -> ConversationInfoResponse:
        return ConversationInfoResponse(
            id=conversation_id, 
            agent_ids=["agent1", "agent2"], 
            project_ids=["project1", "project2"],
            total_messages=100,  
        )

    async def get_conversation_messages(self, conversation_id: str, start: int, limit: int) -> List[MessageResponse]:
        return [
            MessageResponse(
                id=f"mock-message-id-{i}",
                conversation_id=conversation_id,
                content=f"Mock message content {i}",
                sender_id="mock-sender-id",
                type="HUMAN",
                reason=None,
                created_at="2024-01-01T00:00:00Z"
            ) for i in range(start, start + limit)
        ]

    async def regenerate_last_message(self, conversation_id: str) -> MessageResponse:
        # Mocked regenerated message
        return MessageResponse(
            id="mock-message-id-regenerated",
            conversation_id=conversation_id,
            content="Regenerated message content",
            sender_id="system", 
            type="AI_GENERATED",
            reason="Regeneration",
            created_at="2024-01-01T00:00:00Z"
        )

    async def delete_conversation(self, conversation_id: str) -> dict:
        return {"status": "success"}

    async def stop_generation(self, conversation_id: str) -> dict:
        return {"status": "stopped"}
