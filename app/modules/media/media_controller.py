from typing import List, Optional

from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io

from app.core.config_provider import config_provider
from app.modules.media.media_service import (
    MediaService,
    MediaServiceError,
    MediaError,
    create_media_error,
)
from app.modules.media.media_schema import (
    AttachmentUploadResponse,
    AttachmentAccessResponse,
    AttachmentInfo,
)
from app.modules.conversations.access.access_service import ShareChatService
from app.modules.conversations.conversation.conversation_schema import (
    ConversationAccessType,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class MediaController:
    def __init__(self, db: Session, user_id: str, user_email: str):
        self.db = db
        self.user_id = user_id
        self.user_email = user_email
        self.media_service = MediaService(db)
        self.share_chat_service = ShareChatService(db)

    def _check_multimodal_enabled(self):
        """Check if multimodal functionality is enabled"""
        if not config_provider.get_is_multimodal_enabled():
            raise HTTPException(
                status_code=501,
                detail={
                    "error": "Multimodal functionality is currently disabled",
                    "code": "MULTIMODAL_DISABLED",
                },
            )

    async def upload_image(
        self, file: UploadFile, message_id: Optional[str] = None
    ) -> AttachmentUploadResponse:
        """Upload image with feature flag check"""
        self._check_multimodal_enabled()
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")

            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")

            # Upload using media service
            result = await self.media_service.upload_image(
                file=file,
                file_name=file.filename,
                mime_type=file.content_type,
                message_id=message_id,
            )

            logger.info(f"User {self.user_id} uploaded image: {result.id}")
            return result

        except MediaServiceError as e:
            logger.error(f"Media service error: {str(e)}")
            raise create_media_error(
                500, MediaError.PROCESSING_ERROR, "Failed to upload image", str(e)
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading image: {str(e)}")
            raise create_media_error(
                500, MediaError.PROCESSING_ERROR, "An unexpected error occurred", None
            )

    async def upload_document(
        self, file: UploadFile, message_id: Optional[str] = None
    ) -> AttachmentUploadResponse:
        """Upload document with feature flag check"""
        self._check_multimodal_enabled()
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")

            # Upload using media service
            result = await self.media_service.upload_document(
                file=file,
                file_name=file.filename,
                mime_type=file.content_type or "application/octet-stream",
                message_id=message_id,
            )

            logger.info(f"User {self.user_id} uploaded document: {result.id}")
            return result

        except MediaServiceError as e:
            logger.error(f"Media service error: {str(e)}")
            raise create_media_error(
                500, MediaError.PROCESSING_ERROR, "Failed to upload document", str(e)
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading document: {str(e)}")
            raise create_media_error(
                500, MediaError.PROCESSING_ERROR, "An unexpected error occurred", None
            )

    async def upload_document(
        self, file: UploadFile, message_id: Optional[str] = None
    ) -> AttachmentUploadResponse:
        """Upload document with feature flag check"""
        self._check_multimodal_enabled()
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")

            # Upload using media service
            result = await self.media_service.upload_document(
                file=file,
                file_name=file.filename,
                mime_type=file.content_type or "application/octet-stream",
                message_id=message_id,
            )

            logger.info(f"User {self.user_id} uploaded document: {result.id}")
            return result

        except MediaServiceError as e:
            logger.error(f"Media service error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading document: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload document")

    async def get_attachment_access_url(
        self, attachment_id: str, expiration_minutes: int = 60
    ) -> AttachmentAccessResponse:
        """Generate a signed URL for accessing an attachment"""
        try:
            # Get attachment to verify it exists and check permissions
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Check if user has access to the message/conversation containing this attachment
            if attachment.message_id:
                access_granted = await self._check_attachment_access(
                    attachment.message_id
                )
                if not access_granted:
                    raise HTTPException(status_code=403, detail="Access denied")

            # Generate signed URL
            signed_url = await self.media_service.generate_signed_url(
                attachment_id, expiration_minutes
            )

            return AttachmentAccessResponse(
                download_url=signed_url,
                expires_in=expiration_minutes * 60,  # Convert to seconds
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating attachment access URL: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate access URL")

    async def download_attachment(self, attachment_id: str) -> StreamingResponse:
        """Download attachment file directly"""
        try:
            # Get attachment to verify it exists and check permissions
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Check if user has access to the message/conversation containing this attachment
            if attachment.message_id:
                access_granted = await self._check_attachment_access(
                    attachment.message_id
                )
                if not access_granted:
                    raise HTTPException(status_code=403, detail="Access denied")

            # Get the file data from storage
            file_data = await self.media_service.get_attachment_data(attachment_id)

            # Create streaming response
            file_stream = io.BytesIO(file_data)

            # Set appropriate headers
            headers = {
                "Content-Disposition": f'inline; filename="{attachment.file_name}"',
                "Content-Type": attachment.mime_type,
                "Content-Length": str(len(file_data)),
            }

            return StreamingResponse(
                io.BytesIO(file_data), media_type=attachment.mime_type, headers=headers
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading attachment: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to download attachment")

    async def get_attachment_info(self, attachment_id: str) -> AttachmentInfo:
        """Get attachment information"""
        try:
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Check access permissions
            if attachment.message_id:
                access_granted = await self._check_attachment_access(
                    attachment.message_id
                )
                if not access_granted:
                    raise HTTPException(status_code=403, detail="Access denied")

            return AttachmentInfo(
                id=attachment.id,
                attachment_type=attachment.attachment_type,
                file_name=attachment.file_name,
                mime_type=attachment.mime_type,
                file_size=attachment.file_size,
                storage_provider=attachment.storage_provider,
                file_metadata=attachment.file_metadata,
                created_at=attachment.created_at,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting attachment info: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get attachment info")

    async def delete_attachment(self, attachment_id: str) -> dict:
        """Delete an attachment"""
        try:
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Check if user has permission to delete (must be message owner or have write access)
            if attachment.message_id:
                access_granted = await self._check_attachment_access(
                    attachment.message_id, require_write=True
                )
                if not access_granted:
                    raise HTTPException(status_code=403, detail="Access denied")

            success = await self.media_service.delete_attachment(attachment_id)
            if success:
                return {"message": "Attachment deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Attachment not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting attachment: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to delete attachment")

    async def get_message_attachments(self, message_id: str) -> List[AttachmentInfo]:
        """Get all attachments for a message"""
        try:
            # Check access to the message
            access_granted = await self._check_attachment_access(message_id)
            if not access_granted:
                raise HTTPException(status_code=403, detail="Access denied")

            attachments = await self.media_service.get_message_attachments(message_id)
            return attachments

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting message attachments: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to get message attachments"
            )

    async def validate_document_upload(
        self,
        conversation_id: str,
        file_size: int,
        file_name: str,
        mime_type: str,
        async_db,
    ) -> dict:
        """
        Validate if a document can be uploaded without exceeding context limits.

        This should be called BEFORE uploading the actual file to prevent
        wasted processing on files that will be rejected.
        """
        self._check_multimodal_enabled()
        try:
            # Get conversation and model info
            from app.modules.conversations.conversation.conversation_controller import (
                ConversationController,
            )
            from app.modules.intelligence.agents.agents_service import AgentsService
            from app.modules.intelligence.provider.provider_service import (
                ProviderService,
            )
            from app.modules.intelligence.prompts.prompt_service import PromptService
            from app.modules.intelligence.tools.tool_service import ToolService
            from app.modules.intelligence.provider.token_counter import (
                get_token_counter,
            )

            controller = ConversationController(
                self.db, async_db, self.user_id, self.user_email
            )
            conversation_info = await controller.get_conversation_info(conversation_id)

            # Get agent model
            agent_id = (
                conversation_info.agent_ids[0] if conversation_info.agent_ids else None
            )
            if not agent_id:
                raise HTTPException(status_code=400, detail="No agent configured")

            # Get model configuration
            provider_service = ProviderService(self.db, self.user_id)
            tool_service = ToolService(self.db, self.user_id)
            prompt_service = PromptService(self.db)
            agent_service = AgentsService(
                self.db, provider_service, prompt_service, tool_service
            )

            # Get model identifier
            agent_type = await agent_service.validate_agent_id(self.user_id, agent_id)
            if agent_type == "CUSTOM_AGENT":
                custom_agent = await agent_service.custom_agent_service.get_agent_model(
                    agent_id
                )
                model = (
                    provider_service.chat_config.model
                    if custom_agent
                    else "openai/gpt-4o"
                )
            else:
                model = provider_service.chat_config.model

            # Get conversation messages (last 20 for context estimation)
            messages = await controller.get_conversation_messages(
                conversation_id, start=0, limit=20
            )

            # Count tokens
            token_counter = get_token_counter()

            # Extract message content and count tokens
            history_content = []
            attachment_tokens = 0

            for msg in messages:
                if msg.content:
                    history_content.append(f"{msg.type}: {msg.content}")

                # Count attachment tokens
                if msg.has_attachments and msg.attachments:
                    for attachment in msg.attachments:
                        if attachment.file_metadata:
                            token_count = attachment.file_metadata.get("token_count", 0)
                            if token_count:
                                attachment_tokens += token_count

            # Calculate current context usage
            history_tokens = token_counter.count_messages_tokens(history_content, model)
            current_tokens = history_tokens + attachment_tokens

            # Estimate tokens for new file
            estimated_tokens = token_counter.estimate_file_tokens(file_size, mime_type)

            # Get model limit
            context_limit = token_counter.get_context_limit(model)

            # Calculate projected usage
            projected_total = current_tokens + estimated_tokens
            can_upload = projected_total <= context_limit

            response = {
                "can_upload": can_upload,
                "estimated_tokens": estimated_tokens,
                "current_context_usage": current_tokens,
                "model": model,
                "model_context_limit": context_limit,
                "remaining_tokens": max(0, context_limit - current_tokens),
                "projected_total": projected_total,
            }

            if not can_upload:
                response["exceeds_limit"] = True
                response["excess_tokens"] = projected_total - context_limit
                response["excess_percentage"] = round(
                    (response["excess_tokens"] / context_limit) * 100, 2
                )
            else:
                response["exceeds_limit"] = False
                response["usage_after_upload"] = round(
                    (projected_total / context_limit) * 100, 2
                )

            logger.info(
                f"Document validation for conversation {conversation_id}: "
                f"can_upload={can_upload}, projected_total={projected_total}/{context_limit}"
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating document upload: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _check_attachment_access(
        self, message_id: str, require_write: bool = False
    ) -> bool:
        """Check if user has access to the message containing the attachment"""
        try:
            # Get the message and its conversation
            from app.modules.conversations.message.message_model import Message

            message = self.db.query(Message).filter(Message.id == message_id).first()

            if not message:
                return False

            # Check conversation access
            access_level = await self.share_chat_service.check_conversation_access(
                message.conversation_id, self.user_email
            )

            if access_level == ConversationAccessType.NOT_FOUND:
                return False

            if require_write and access_level == ConversationAccessType.READ:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking attachment access: {str(e)}")
            return False

    async def test_multimodal_functionality(self, attachment_id: str) -> dict:
        """Test multimodal with feature flag check"""
        self._check_multimodal_enabled()
        try:
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Check access permissions
            if attachment.message_id:
                access_granted = await self._check_attachment_access(
                    attachment.message_id
                )
                if not access_granted:
                    raise HTTPException(status_code=403, detail="Access denied")

            # Test multimodal functionality
            result = await self.media_service.test_multimodal_functionality(
                attachment_id
            )
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error testing multimodal functionality: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to test multimodal functionality"
            )
