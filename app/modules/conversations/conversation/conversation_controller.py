from typing import AsyncGenerator, List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.conversations.conversation.conversation_schema import (
    ConversationInfoResponse,
    CreateConversationRequest,
    CreateConversationResponse,
)
from app.modules.conversations.conversation.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
    ConversationServiceError,
)
from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import (
    MessageRequest,
    MessageResponse,
)


class ConversationController:
    def __init__(self, db: Session):
        self.service = ConversationService.create(db)

    async def create_conversation(
        self, conversation: CreateConversationRequest, user_id: str
    ) -> CreateConversationResponse:
        try:
            conversation_id, message = await self.service.create_conversation(
                conversation, user_id
            )
            return CreateConversationResponse(
                message=message, conversation_id=conversation_id
            )
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_conversation(self, conversation_id: str, user_id: str) -> dict:
        try:
            return await self.service.delete_conversation(conversation_id, user_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_conversation_info(
        self, conversation_id: str, user_id: str
    ) -> ConversationInfoResponse:
        try:
            return await self.service.get_conversation_info(conversation_id, user_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_conversation_messages(
        self, conversation_id: str, start: int, limit: int, user_id: str
    ) -> List[MessageResponse]:
        try:
            return await self.service.get_conversation_messages(
                conversation_id, start, limit, user_id
            )
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def post_message(
        self, conversation_id: str, message: MessageRequest, user_id: str
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self.service.store_message(
                conversation_id, message, MessageType.HUMAN, user_id
            ):
                yield chunk
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def regenerate_last_message(
        self, conversation_id: str, user_id: str
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self.service.regenerate_last_message(
                conversation_id, user_id
            ):
                yield chunk
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def stop_generation(self, conversation_id: str, user_id: str) -> dict:
        try:
            return await self.service.stop_generation(conversation_id, user_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
