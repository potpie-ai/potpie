from typing import List
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from fastapi import Depends

from app.core.database import get_db
from app.modules.conversations.conversation.conversation_schema import ConversationInfoResponse, ConversationResponse, CreateConversationRequest, CreateConversationResponse
from app.modules.conversations.conversation.conversation_service import ConversationService
from app.modules.conversations.message.message_schema import MessageRequest, MessageResponse



class ConversationController:
    def __init__(self, db: Session):
        self.service = ConversationService(db)

    async def create_conversation(self, conversation: CreateConversationRequest) -> CreateConversationResponse:
        conversation_id, message = await self.service.create_conversation(conversation)
        if isinstance(message, str):
            return CreateConversationResponse(message=message, conversation_id=conversation_id)
        else:
            raise ValueError("Message returned from service is not a string")

    async def get_conversation(self, conversation_id: str) -> ConversationResponse:
        # Example logic to retrieve a conversation
        return await self.service.get_conversation(conversation_id)

    async def get_conversation_info(self, conversation_id: str) -> ConversationInfoResponse:
        # Example logic to retrieve conversation info
        return await self.service.get_conversation_info(conversation_id)

    async def get_conversation_messages(self, conversation_id: str, start: int, limit: int) -> List[MessageResponse]:
        # Example logic to retrieve conversation messages
        return await self.service.get_conversation_messages(conversation_id, start, limit)

    async def post_message(self, conversation_id: str, message: MessageRequest, user_id: str):
        stored_message = await self.service.store_message(conversation_id, message, user_id)
        message_stream = self.service.message_stream(conversation_id)
        return StreamingResponse(message_stream, media_type="text/event-stream")

    async def regenerate_last_message(self, conversation_id: str) -> MessageResponse:
        return await self.service.regenerate_last_message(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> dict:
        return await self.service.delete_conversation(conversation_id)

    async def stop_generation(self, conversation_id: str) -> dict:
        return await self.service.stop_generation(conversation_id)