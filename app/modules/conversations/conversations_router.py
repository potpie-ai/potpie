from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
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
from .message.message_schema import MessageRequest, MessageResponse, RegenerateRequest
from app.modules.conversations.access.access_service import ShareChatService, ShareChatServiceError
from app.modules.conversations.access.access_schema import ShareChatRequest, ShareChatResponse, SharedChatResponse

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
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
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
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
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
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
        return await controller.get_conversation_messages(conversation_id, start, limit)

    @staticmethod
    @router.post("/conversations/{conversation_id}/message/")
    async def post_message(
        conversation_id: str,
        message: MessageRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        if (
            message.content == ""
            or message.content is None
            or message.content.isspace()
        ):
            raise HTTPException(
                status_code=400, detail="Message content cannot be empty"
            )

        user_id = user["user_id"]
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
        message_stream = controller.post_message(conversation_id, message)
        return StreamingResponse(message_stream, media_type="text/event-stream")

    @staticmethod
    @router.post(
        "/conversations/{conversation_id}/regenerate/", response_model=MessageResponse
    )
    async def regenerate_last_message(
        conversation_id: str,
        request: RegenerateRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
        return StreamingResponse(
            controller.regenerate_last_message(conversation_id, request.node_ids),
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
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
        return await controller.delete_conversation(conversation_id)

    @staticmethod
    @router.post("/conversations/{conversation_id}/stop/", response_model=dict)
    async def stop_generation(
        conversation_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
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
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
        return await controller.rename_conversation(conversation_id, request.title)

@router.post("/conversations/share", response_model=ShareChatResponse, status_code=201)
async def share_chat(
    request: ShareChatRequest,
    db: Session = Depends(get_db),
):
    service = ShareChatService(db)
    try:
        shared_conversation = await service.share_chat(request.conversation_id, request.recipientEmails)
        return ShareChatResponse(
            message="Chat shared successfully!",
            sharedID=shared_conversation
        )
    except ShareChatServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))

