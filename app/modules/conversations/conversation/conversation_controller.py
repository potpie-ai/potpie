from typing import AsyncGenerator, List
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.modules.conversations.conversation.conversation_schema import ConversationInfoResponse, ConversationResponse, CreateConversationRequest, CreateConversationResponse
from app.modules.conversations.conversation.conversation_service import ConversationService
from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import MessageRequest, MessageResponse

class ConversationController:
    def __init__(self, db: Session):
        self.service = ConversationService(db)

    async def create_conversation(self, conversation: CreateConversationRequest) -> CreateConversationResponse:
        conversation_id, message = await self.service.create_conversation(conversation)
        return CreateConversationResponse(message=message, conversation_id=conversation_id)

    async def get_conversation(self, conversation_id: str) -> ConversationResponse:
        return await self.service.get_conversation(conversation_id)
    
    async def delete_conversation(self, conversation_id: str) -> dict:
        return await self.service.delete_conversation(conversation_id)
    
    async def get_conversation_info(self, conversation_id: str) -> ConversationInfoResponse:
        return await self.service.get_conversation_info(conversation_id)

    async def get_conversation_messages(self, conversation_id: str, start: int, limit: int) -> List[MessageResponse]:
        return await self.service.get_conversation_messages(conversation_id, start, limit)
    
    async def post_message(self, conversation_id: str, message: MessageRequest, user_id: str) -> AsyncGenerator[str, None]:
        # Store the message in the database with type MessageType.Human
        stored_message = await self.service.store_message(conversation_id, message,MessageType.HUMAN, user_id)
        
        # Stream the DuckDuckGo response back to the user
        async for chunk in self.service.message_stream(conversation_id, stored_message.content):
            yield chunk  # This yields each part of the stream as an async generator

    
    async def regenerate_last_message(self, conversation_id: str) -> MessageResponse:
        return await self.service.regenerate_last_message(conversation_id)
    
    async def stop_generation(self, conversation_id: str) -> dict:
        return await self.service.stop_generation(conversation_id)