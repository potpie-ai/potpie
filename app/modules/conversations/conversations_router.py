import json
from typing import Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.conversations.access.access_schema import (
    RemoveAccessRequest,
    ShareChatRequest,
    ShareChatResponse,
)
from app.modules.conversations.access.access_service import (
    ShareChatService,
    ShareChatServiceError,
)
from app.modules.conversations.conversation.conversation_controller import (
    ConversationController,
)
from app.modules.usage.usage_service import UsageService


from .conversation.conversation_schema import (
    ConversationInfoResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    RenameConversationRequest,
)
from .message.message_schema import MessageRequest, MessageResponse, RegenerateRequest

router = APIRouter()


async def get_stream(data_stream: AsyncGenerator[Any, None]):
    async for chunk in data_stream:
        yield json.dumps(chunk.dict())


class ConversationAPI:
    @staticmethod
    @router.post("/conversations/", response_model=CreateConversationResponse)
    async def create_conversation(
        conversation: CreateConversationRequest,
        hidden: bool = Query(
            False, description="Whether to hide this conversation from the web UI"
        ),
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        checked = await UsageService.check_usage_limit(user_id)
        if not checked:
            raise HTTPException(
                status_code=402,
                detail="Subscription required to create a conversation.",
            )
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
        return await controller.create_conversation(conversation, hidden)

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
        stream: bool = Query(True, description="Whether to stream the response"),
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
        checked = await UsageService.check_usage_limit(user_id)
        if not checked:
            raise HTTPException(
                status_code=402,
                detail="Subscription required to create a conversation.",
            )
        controller = ConversationController(db, user_id, user_email)
        message_stream = controller.post_message(conversation_id, message, stream)
        if stream:
            return StreamingResponse(
                get_stream(message_stream), media_type="text/event-stream"
            )
        else:
            # TODO: fix this, add types. In below stream we have only one output.
            async for chunk in message_stream:
                return chunk

    @staticmethod
    @router.post(
        "/conversations/{conversation_id}/regenerate/", response_model=MessageResponse
    )
    async def regenerate_last_message(
        conversation_id: str,
        request: RegenerateRequest,
        stream: bool = Query(True, description="Whether to stream the response"),
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        checked = await UsageService.check_usage_limit(user_id)
        if not checked:
            raise HTTPException(
                status_code=402,
                detail="Subscription required to create a conversation.",
            )
        user_email = user["email"]
        controller = ConversationController(db, user_id, user_email)
        message_stream = controller.regenerate_last_message(
            conversation_id, request.node_ids, stream
        )
        if stream:
            return StreamingResponse(
                get_stream(message_stream), media_type="text/event-stream"
            )
        else:
            async for chunk in message_stream:
                return chunk

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
    user=Depends(AuthService.check_auth),
):
    user_id = user["user_id"]
    service = ShareChatService(db)
    try:
        shared_conversation = await service.share_chat(
            request.conversation_id,
            user_id,
            request.recipientEmails,
            request.visibility,
        )
        return ShareChatResponse(
            message="Chat shared successfully!", sharedID=shared_conversation
        )
    except ShareChatServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/{conversation_id}/shared-emails", response_model=List[str])
async def get_shared_emails(
    conversation_id: str,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    user_id = user["user_id"]
    service = ShareChatService(db)
    shared_emails = await service.get_shared_emails(conversation_id, user_id)
    return shared_emails


@router.delete("/conversations/{conversation_id}/access")
async def remove_access(
    conversation_id: str,
    request: RemoveAccessRequest,
    user: str = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict:
    """Remove access for specified emails from a conversation."""
    share_service = ShareChatService(db)
    current_user_id = user["user_id"]
    try:
        await share_service.remove_access(
            conversation_id=conversation_id,
            user_id=current_user_id,
            emails_to_remove=request.emails,
        )
        return {"message": "Access removed successfully"}
    except ShareChatServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
