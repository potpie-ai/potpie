import json
from typing import Any, AsyncGenerator, List, Optional, Union, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, get_async_db
from app.modules.auth.auth_service import AuthService
from app.modules.utils.logger import setup_logger, log_context
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
from app.modules.media.media_service import MediaService

from .conversation.conversation_schema import (
    ConversationInfoResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    RenameConversationRequest,
    ActiveSessionResponse,
    ActiveSessionErrorResponse,
    TaskStatusResponse,
    TaskStatusErrorResponse,
)
from .message.message_schema import MessageRequest, MessageResponse, RegenerateRequest
from .session.session_service import SessionService
from app.modules.users.user_schema import UserConversationListResponse

router = APIRouter()
logger = setup_logger(__name__)


from app.modules.conversations.utils.conversation_routing import (
    normalize_run_id,
    ensure_unique_run_id,
    redis_stream_generator,
    start_celery_task_and_stream,
)


async def get_stream(data_stream: AsyncGenerator[Any, None]):
    async for chunk in data_stream:
        yield json.dumps(chunk.dict())


class ConversationAPI:
    @staticmethod
    @router.get(
        "/conversations",
        response_model=List[UserConversationListResponse],
        description="Get a list of conversations for the current user with sorting options.",
    )
    async def get_conversations_for_user(
        user=Depends(AuthService.check_auth),
        start: int = Query(0, ge=0),
        limit: int = Query(10, ge=1),
        sort: Literal["updated_at", "created_at"] = Query(
            "updated_at", description="Field to sort by"
        ),
        order: Literal["asc", "desc"] = Query("desc", description="Direction of sort"),
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
    ):
        """Get a list of conversations for the current user with sorting options."""
        user_id = user["user_id"]
        user_email = user["email"]
        controller = ConversationController(db, async_db, user_id, user_email)
        return await controller.get_conversations_for_user(start, limit, sort, order)

    @staticmethod
    @router.post("/conversations", response_model=CreateConversationResponse)
    async def create_conversation(
        conversation: CreateConversationRequest,
        hidden: bool = Query(
            False, description="Whether to hide this conversation from the web UI"
        ),
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
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
        controller = ConversationController(db, async_db, user_id, user_email)
        return await controller.create_conversation(conversation, hidden)

    @staticmethod
    @router.get(
        "/conversations/{conversation_id}/info/",
        response_model=ConversationInfoResponse,
    )
    async def get_conversation_info(
        conversation_id: str,
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        user_email = user["email"]

        controller = ConversationController(db, async_db, user_id, user_email)

        try:
            result = await controller.get_conversation_info(conversation_id)
            return result
        except Exception as e:
            logger.error(
                f"Error in get_conversation_info for {conversation_id}: {str(e)}",
                exc_info=True,
            )
            raise

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
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        user_email = user["email"]

        controller = ConversationController(db, async_db, user_id, user_email)

        try:
            result = await controller.get_conversation_messages(
                conversation_id, start, limit
            )
            return result
        except Exception as e:
            logger.error(
                f"Error in get_conversation_messages for {conversation_id}: {str(e)}",
                exc_info=True,
            )
            raise

    @staticmethod
    @router.post("/conversations/{conversation_id}/message/")
    async def post_message(
        conversation_id: str,
        content: str = Form(...),
        node_ids: Optional[str] = Form(None),
        images: Optional[List[UploadFile]] = File(None),
        stream: bool = Query(True, description="Whether to stream the response"),
        session_id: Optional[str] = Query(
            None, description="Session ID for reconnection"
        ),
        prev_human_message_id: Optional[str] = Query(
            None, description="Previous human message ID for deterministic session ID"
        ),
        cursor: Optional[str] = Query(None, description="Stream cursor for replay"),
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        # Validate message content
        if content == "" or content is None or content.isspace():
            raise HTTPException(
                status_code=400, detail="Message content cannot be empty"
            )

        user_id = user["user_id"]
        user_email = user["email"]

        # Set up logging context with domain IDs
        with log_context(conversation_id=conversation_id, user_id=user_id):
            checked = await UsageService.check_usage_limit(user_id)
            if not checked:
                raise HTTPException(
                    status_code=402,
                    detail="Subscription required to create a conversation.",
                )

            # Process images if present
            attachment_ids = []
            if images:
                media_service = MediaService(db)
                for _i, image in enumerate(images):
                    # Check if image has content by checking filename and content_type
                    if image.filename and image.content_type:
                        try:
                            # Read file data first and pass as bytes to avoid UploadFile issues
                            file_content = await image.read()
                            upload_result = await media_service.upload_image(
                                file=file_content,
                                file_name=image.filename,
                                mime_type=image.content_type,
                                message_id=None,  # Will be linked after message creation
                            )
                            attachment_ids.append(upload_result.id)
                        except Exception as e:
                            logger.exception(
                                "Failed to upload image",
                                filename=image.filename,
                                conversation_id=conversation_id,
                                user_id=user_id,
                            )
                            # Clean up any successfully uploaded attachments
                            for uploaded_id in attachment_ids:
                                try:
                                    await media_service.delete_attachment(uploaded_id)
                                except Exception as cleanup_exc:
                                    logger.warning(
                                        f"Failed to cleanup attachment {uploaded_id} after image upload error: {str(cleanup_exc)}",
                                        conversation_id=conversation_id,
                                        user_id=user_id,
                                        attachment_id=uploaded_id,
                                    )
                            raise HTTPException(
                                status_code=400,
                                detail=f"Failed to upload image {image.filename}: {str(e)}",
                            ) from e

            # Parse node_ids if provided
            parsed_node_ids = None
            if node_ids:
                try:
                    parsed_node_ids = json.loads(node_ids)
                except json.JSONDecodeError as err:
                    raise HTTPException(
                        status_code=400, detail="Invalid node_ids format"
                    ) from err

            # Create message request
            message = MessageRequest(
                content=content,
                node_ids=parsed_node_ids,
                attachment_ids=attachment_ids if attachment_ids else None,
            )

            controller = ConversationController(db, async_db, user_id, user_email)

            if not stream:
                # Non-streaming behavior unchanged
                message_stream = controller.post_message(
                    conversation_id, message, stream
                )
                async for chunk in message_stream:
                    return chunk

            # Streaming with session management
            run_id = normalize_run_id(
                conversation_id, user_id, session_id, prev_human_message_id
            )

            # For fresh requests without cursor, ensure we get a unique stream
            if not cursor:
                run_id = ensure_unique_run_id(conversation_id, run_id)

            # Use parsed node_ids
            node_ids_list = parsed_node_ids or []

            # Start background task and return streaming response
            return start_celery_task_and_stream(
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                query=content,
                agent_id=None,
                node_ids=node_ids_list,
                attachment_ids=attachment_ids or [],
                cursor=cursor,
            )

    @staticmethod
    @router.post("/conversations/{conversation_id}/regenerate/")
    async def regenerate_last_message(
        conversation_id: str,
        request: RegenerateRequest,
        stream: bool = Query(True, description="Whether to stream the response"),
        session_id: Optional[str] = Query(
            None, description="Session ID for reconnection"
        ),
        prev_human_message_id: Optional[str] = Query(
            None, description="Previous human message ID for deterministic session ID"
        ),
        cursor: Optional[str] = Query(None, description="Stream cursor for replay"),
        background: bool = Query(
            True, description="Use background execution (recommended)"
        ),
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
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

        if not stream or not background:
            # Fallback to existing direct execution for non-streaming or explicit direct mode
            controller = ConversationController(db, async_db, user_id, user_email)
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

        # NEW: Background execution with session management
        controller = ConversationController(db, async_db, user_id, user_email)

        # Generate deterministic run_id
        run_id = normalize_run_id(
            conversation_id, user_id, session_id, prev_human_message_id
        )

        # For fresh requests without cursor, ensure we get a unique stream
        if not cursor:
            run_id = ensure_unique_run_id(conversation_id, run_id)

        # Extract attachment IDs from last human message
        try:
            # Get last human message to extract attachments
            last_human_message = await controller.get_last_human_message(
                conversation_id
            )
            attachment_ids = []
            if last_human_message and last_human_message.has_attachments:
                # Use media service to get attachments instead of accessing relationship directly
                # This avoids SQLAlchemy async lazy-loading issues
                try:
                    media_service = MediaService(db)
                    attachments = await media_service.get_message_attachments(
                        last_human_message.id, include_download_urls=False
                    )
                    attachment_ids = [att.id for att in attachments]
                except Exception as e:
                    logger.warning(
                        f"Failed to retrieve attachments for message {last_human_message.id}: {e}"
                    )
                    attachment_ids = []
        except Exception as e:
            logger.error(f"Failed to get last human message for regenerate: {str(e)}")
            attachment_ids = []

        # Start background regenerate task
        from app.celery.tasks.agent_tasks import execute_regenerate_background
        from app.modules.conversations.utils.redis_streaming import RedisStreamManager

        redis_manager = RedisStreamManager()
        # Set initial "queued" status before starting the task
        redis_manager.set_task_status(conversation_id, run_id, "queued")

        # Publish a queued event so the client knows the task is accepted
        redis_manager.publish_event(
            conversation_id,
            run_id,
            "queued",
            {
                "status": "queued",
                "message": "Regeneration task queued for processing",
            },
        )

        task_result = execute_regenerate_background.delay(
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            node_ids=request.node_ids or [],
            attachment_ids=attachment_ids,
        )

        # Store the Celery task ID for later revocation
        redis_manager.set_task_id(conversation_id, run_id, task_result.id)
        logger.info(
            f"Started regenerate task {task_result.id} for {conversation_id}:{run_id}"
        )

        # Wait for background task to start (with health check)
        # Increased timeout to 30 seconds to handle queued tasks
        task_started = redis_manager.wait_for_task_start(
            conversation_id, run_id, timeout=30
        )

        if not task_started:
            logger.warning(
                f"Background regenerate task failed to start within 30s for {conversation_id}:{run_id} - may still be queued"
            )
            # Don't fail - the stream consumer will wait up to 120 seconds

        # Return Redis stream response using shared function
        return StreamingResponse(
            redis_stream_generator(conversation_id, run_id, cursor),
            media_type="text/event-stream",
        )

    @staticmethod
    @router.delete("/conversations/{conversation_id}/", response_model=dict)
    async def delete_conversation(
        conversation_id: str,
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        user_email = user["email"]
        controller = ConversationController(db, async_db, user_id, user_email)
        return await controller.delete_conversation(conversation_id)

    @staticmethod
    @router.post("/conversations/{conversation_id}/stop/", response_model=dict)
    async def stop_generation(
        conversation_id: str,
        session_id: Optional[str] = Query(None, description="Session ID to stop"),
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        user_email = user["email"]
        controller = ConversationController(db, async_db, user_id, user_email)
        return await controller.stop_generation(conversation_id, session_id)

    @staticmethod
    @router.patch("/conversations/{conversation_id}/rename/", response_model=dict)
    async def rename_conversation(
        conversation_id: str,
        request: RenameConversationRequest,
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        user_email = user["email"]
        controller = ConversationController(db, async_db, user_id, user_email)
        return await controller.rename_conversation(conversation_id, request.title)

    @staticmethod
    @router.get("/conversations/{conversation_id}/active-session")
    async def get_active_session(
        conversation_id: str,
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ) -> Union[ActiveSessionResponse, ActiveSessionErrorResponse]:
        """Get active session information for a conversation"""
        user_id = user["user_id"]
        user_email = user["email"]

        # Verify user has access to conversation
        controller = ConversationController(db, async_db, user_id, user_email)
        try:
            await controller.get_conversation_info(conversation_id)
        except Exception as e:
            logger.error(f"Access denied for conversation {conversation_id}: {str(e)}")
            raise HTTPException(status_code=403, detail="Access denied to conversation")

        # Get session information
        session_service = SessionService()
        result = session_service.get_active_session(conversation_id)

        # Return appropriate HTTP status based on result type
        if isinstance(result, ActiveSessionErrorResponse):
            raise HTTPException(status_code=404, detail=result.dict())

        return result

    @staticmethod
    @router.get("/conversations/{conversation_id}/task-status")
    async def get_task_status(
        conversation_id: str,
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ) -> Union[TaskStatusResponse, TaskStatusErrorResponse]:
        """Get background task status for a conversation"""
        user_id = user["user_id"]
        user_email = user["email"]

        # Verify user has access to conversation
        controller = ConversationController(db, async_db, user_id, user_email)
        try:
            await controller.get_conversation_info(conversation_id)
        except Exception as e:
            logger.error(f"Access denied for conversation {conversation_id}: {str(e)}")
            raise HTTPException(status_code=403, detail="Access denied to conversation")

        # Get task status information
        session_service = SessionService()
        result = session_service.get_task_status(conversation_id)

        # Return appropriate HTTP status based on result type
        if isinstance(result, TaskStatusErrorResponse):
            raise HTTPException(status_code=404, detail=result.dict())

        return result

    @staticmethod
    @router.post("/conversations/{conversation_id}/resume/{session_id}")
    async def resume_session(
        conversation_id: str,
        session_id: str,
        cursor: Optional[str] = Query(
            "0-0", description="Stream cursor position to resume from"
        ),
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        """Resume streaming from an existing session"""
        user_id = user["user_id"]
        user_email = user["email"]

        # Verify user has access to conversation
        controller = ConversationController(db, async_db, user_id, user_email)
        try:
            await controller.get_conversation_info(conversation_id)
        except Exception as e:
            logger.error(f"Access denied for conversation {conversation_id}: {str(e)}")
            raise HTTPException(status_code=403, detail="Access denied to conversation")

        # Verify the session exists in Redis
        from app.modules.conversations.utils.redis_streaming import RedisStreamManager

        redis_manager = RedisStreamManager()

        # Check if the session stream exists
        stream_key = redis_manager.stream_key(conversation_id, session_id)
        if not redis_manager.redis_client.exists(stream_key):
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found or expired"
            )

        # Check if there's a task status for this session
        task_status = redis_manager.get_task_status(conversation_id, session_id)
        logger.info(
            f"Resuming session {session_id} with status: {task_status}, cursor: {cursor}"
        )

        # Return Redis stream response starting from cursor
        return StreamingResponse(
            redis_stream_generator(conversation_id, session_id, cursor),
            media_type="text/event-stream",
        )


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
