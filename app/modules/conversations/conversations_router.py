from typing import List

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.conversations.conversation.conversation_controller import (
    ConversationController,
)

from .conversation.conversation_schema import (
    ConversationInfoResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    RenameConversationRequest,
)
from .message.message_schema import MessageRequest, MessageResponse

router = APIRouter()


class ConversationAPI:
    @staticmethod
    @router.post("/conversations/", response_model=CreateConversationResponse)
    async def create_conversation(
        conversation: CreateConversationRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        return await controller.create_conversation(conversation)

    @staticmethod
    @router.get(
        "/conversations/{conversation_id}/info/",
        response_model=ConversationInfoResponse,
    )
    async def get_conversation_info(
        conversation_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        return await controller.get_conversation_info(conversation_id)

    @staticmethod
    @router.get(
        "/conversations/{conversation_id}/messages/",
        response_model=List[MessageResponse],
    )
    async def get_conversation_messages(
        conversation_id: str,
        start: int = Query(0, ge=0),
        limit: int = Query(10, ge=1),
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        return await controller.get_conversation_messages(conversation_id, start, limit)

    @staticmethod
    @router.post("/conversations/{conversation_id}/message/")
    async def post_message(
        conversation_id: str,
        message: MessageRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        message_stream = controller.post_message(conversation_id, message)
        return StreamingResponse(message_stream, media_type="text/event-stream")

    @staticmethod
    @router.post(
        "/conversations/{conversation_id}/regenerate/", response_model=MessageResponse
    )
    async def regenerate_last_message(
        conversation_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        return StreamingResponse(
            controller.regenerate_last_message(conversation_id),
            media_type="text/event-stream",
        )

    @staticmethod
    @router.delete("/conversations/{conversation_id}/", response_model=dict)
    async def delete_conversation(
        conversation_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        return await controller.delete_conversation(conversation_id)

    @staticmethod
    @router.post("/conversations/{conversation_id}/stop/", response_model=dict)
    async def stop_generation(
        conversation_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        return await controller.stop_generation(conversation_id)

    @staticmethod
    @router.patch("/conversations/{conversation_id}/rename/", response_model=dict)
    async def rename_conversation(
        conversation_id: str,
        request: RenameConversationRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ConversationController(db, user_id)
        return await controller.rename_conversation(conversation_id, request.title)
