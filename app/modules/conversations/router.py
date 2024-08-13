from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from sqlalchemy.orm import Session
from app.core.database import get_db
import asyncio

from .conversation.schema import (
    CreateConversationRequest, 
    CreateConversationResponse, 
    ConversationResponse, 
    ConversationInfoResponse, 
)

from .message.schema import (
    MessageRequest,
    MessageResponse,
    MessageMetadata,
    MessageContentUpdate
)

router = APIRouter()


class ConversationAPI:

    @staticmethod
    @router.post("/conversations/", response_model=CreateConversationResponse)
    def create_conversation(
        conversation: CreateConversationRequest,
        db: Session = Depends(get_db)
    ):
        # Mocked data instead of actual logic
        conversation_id = "mock-conversation-id"
        return CreateConversationResponse(message="project setup complete", conversation_id=conversation_id)

    @staticmethod
    @router.get("/conversations/{conversation_id}/", response_model=ConversationResponse)
    def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
        
        return ConversationResponse(
            id="mock-conversation-id",
            user_id="mock-user-id",
            title="Mock Conversation Title",
            status="active",
            project_ids=["project1", "project2"],
            agent_ids=["agent1", "agent2"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            messages=[]
        )

    @staticmethod
    @router.get("/conversations/{conversation_id}/info/", response_model=ConversationInfoResponse)
    def get_conversation_info(
        conversation_id: str,
        db: Session = Depends(get_db)
    ):
        # Assume 100 total messages for the mock
        return ConversationInfoResponse(
            id="mock-conversation-id", 
            agent_ids=["agent1", "agent2"], 
            project_ids=["project1", "project2"],
            total_messages=100,  # Mocked total message count
        )

    @staticmethod
    @router.get("/conversations/{conversation_id}/messages/", response_model=List[MessageResponse])
    def get_conversation_messages(
        conversation_id: str,
        start: int = Query(0, ge=0),  # Start index, default is 0
        limit: int = Query(10, ge=1),  # Number of items to return, default is 10
        db: Session = Depends(get_db)
    ):
        # Mocked data instead of actual logic
        messages = [
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
        return messages

    @staticmethod
    @router.post("/conversations/{conversation_id}/message/")
    async def post_message(
        conversation_id: str,
        message: MessageRequest,
        db: Session = Depends(get_db)
    ):
        async def generate_message_content():
            # Simulate content generation in chunks
            content_parts = [
                f"{message.content} (part 1)",
                f"{message.content} (part 2)",
                f"{message.content} (part 3)"
            ]

            for part in content_parts:
                await asyncio.sleep(1)  # Simulate delay in content generation
                yield part  # Yield plain text content

        async def message_stream():
            # First, yield the message metadata (if necessary)
            metadata = f"Message ID: mock-message-id, Conversation ID: {conversation_id}, Type: {"AI_GENERATED"}, Reason: {"STREAM"}\n"
            yield metadata

            # Then, stream the content updates
            async for content_update in generate_message_content():
                yield content_update

        return StreamingResponse(message_stream(), media_type="text/plain")

    @staticmethod
    @router.post("/conversations/{conversation_id}/regenerate/", response_model=MessageResponse)
    async def regenerate_last_message(
        conversation_id: str,
        db: Session = Depends(get_db)
    ):
        # Mocked logic for regenerating the last message in the conversation
        await asyncio.sleep(1)  # Simulate processing delay

        # Mocked regenerated message
        regenerated_message = MessageResponse(
            id="mock-message-id-regenerated",
            conversation_id=conversation_id,
            content="Regenerated message content",
            sender_id="system",  # Assuming it's AI generated
            type="AI_GENERATED",
            reason="Regeneration",
            created_at="2024-01-01T00:00:00Z"
        )

        return regenerated_message

    @staticmethod
    @router.delete("/conversations/{conversation_id}/", response_model=dict)
    def delete_conversation(
        conversation_id: str, 
        db: Session = Depends(get_db)
    ):
        # Mocked success response
        return {"status": "success"}