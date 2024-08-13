import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator
from sqlalchemy.orm import Session

from app.modules.conversations.message.schema import MessageRequest
from .schema import CreateConversationRequest
from app.modules.conversations.message.model import Message
from app.modules.intelligence.agents.agent_registry import get_agent

class ConversationService:

    def store_message(self, conversation_id: str, message: MessageRequest, db: Session, user_id: str):
        # Assume `user_id` is passed from the controller based on the authenticated user
        # Create a new Message object
        new_message = Message(
            conversation_id=conversation_id,
            content=message.content,
            # sender_id=user_id,  # Attach the sender_id from the authenticated user
            type="AI_GENERATED",  # Assume "HUMAN" as the message type, could be dynamic
            created_at=datetime.now()# Set the current timestamp
        )
        
        # Add to the session and commit to store in the database
        db.add(new_message)
        db.commit()
        db.refresh(new_message)  # Refresh to get the newly generated ID, if needed
        
        return new_message

    async def generate_message_content(self):
        # Simulate content generation in chunks
        content_parts = [
            "string (part 1)",
            "string (part 2)",
            "string (part 3)"
        ]

        for part in content_parts:
            await asyncio.sleep(1)  # Simulate delay in content generation
            yield f"data: {json.dumps({'content': part})}\n\n"  # Yield each chunk

    async def message_stream(self, conversation_id: str):
        # First, yield the message metadata as JSON
        metadata = {
            "message_id": "mock-message-id",
            "conversation_id": conversation_id,
            "type": "AI_GENERATED",
            "reason": "STREAM"
        }
        yield f"data: {json.dumps(metadata)}\n\n"  # Stream metadata first

        # Then, stream the content updates
        async for content_update in self.generate_message_content():
            yield content_update
    
    def create_conversation(self, conversation: CreateConversationRequest):
        # Generate a mock conversation ID (could be replaced with DB logic if needed)
        conversation_id = "mock-conversation-id"
        
        print("conversation title",conversation.title)

        # Perform a search using the DuckDuckGo agent
        agent = get_agent("langchain_duckduckgo")
        search_result = agent.run("Search for momentum")

        # Return the generated ID and search result as the first message
        return conversation_id, search_result
