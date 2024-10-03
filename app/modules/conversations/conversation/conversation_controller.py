from typing import AsyncGenerator, List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.conversations.conversation.conversation_schema import (
    ConversationInfoResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    ConversationAccessType
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
    NodeContext,
)


class ConversationController:
    def __init__(self, db: Session, user_id: str, user_email: str):
        self.service = ConversationService.create(db, user_id)
        self.user_id = user_id
        self.user_email = user_email

    async def create_conversation(
        self, conversation: CreateConversationRequest
    ) -> CreateConversationResponse:
        try:
            conversation_id, message = await self.service.create_conversation(
                conversation, self.user_id
            )
            return CreateConversationResponse(
                message=message, conversation_id=conversation_id
            )
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_conversation(self, conversation_id: str) -> dict:
        access_level = await self.service.check_conversation_access(conversation_id, self.user_email)
        if access_level == ConversationAccessType.READ:
            raise HTTPException(status_code=403, detail="Access denied.")
        try:
            return await self.service.delete_conversation(conversation_id, self.user_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_conversation_info(
        self, conversation_id: str
    ) -> ConversationInfoResponse:
        access_level = await self.service.check_conversation_access(conversation_id, self.user_email)
        if access_level == ConversationAccessType.NOT_FOUND:
            raise HTTPException(status_code=403, detail="Not Found")
        
        try:
            return await self.service.get_conversation_info(
                conversation_id, self.user_id
            )
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_conversation_messages(
        self, conversation_id: str, start: int, limit: int
    ) -> List[MessageResponse]:
        access_level = await self.service.check_conversation_access(conversation_id, self.user_email)
        if access_level == ConversationAccessType.NOT_FOUND:
            raise HTTPException(status_code=403, detail="Not Found.")
        try:
            return await self.service.get_conversation_messages(
                conversation_id, start, limit, self.user_id
            )
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def post_message(
        self, conversation_id: str, message: MessageRequest
    ) -> AsyncGenerator[str, None]:
        access_level = await self.service.check_conversation_access(conversation_id, self.user_email)
        if access_level == ConversationAccessType.READ:
            raise HTTPException(status_code=403, detail="Read Only.")
        try:
            async for chunk in self.service.store_message(
                conversation_id, message, MessageType.HUMAN, self.user_id
            ):
                yield chunk
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def regenerate_last_message(
        self, conversation_id: str, node_ids: List[NodeContext] = []
    ) -> AsyncGenerator[str, None]:
        access_level = await self.service.check_conversation_access(conversation_id, self.user_email)
        if access_level == ConversationAccessType.READ:
            raise HTTPException(status_code=403, detail="Read Only.")
        try:
            async for chunk in self.service.regenerate_last_message(
                conversation_id, self.user_id, node_ids
            ):
                yield chunk
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def stop_generation(self, conversation_id: str) -> dict:
        try:
            return await self.service.stop_generation(conversation_id, self.user_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def rename_conversation(self, conversation_id: str, new_title: str) -> dict:
        access_level = await self.service.check_conversation_access(conversation_id, self.user_email)
        if access_level == ConversationAccessType.READ:
            raise HTTPException(status_code=403, detail="Read Only.")
        try:
            return await self.service.rename_conversation(
                conversation_id, new_title, self.user_id
            )
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ConversationServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
