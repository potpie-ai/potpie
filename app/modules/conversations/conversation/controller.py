from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.modules.conversations.message.schema import MessageRequest

from .service import ConversationService
from .schema import CreateConversationRequest, CreateConversationResponse

class ConversationController:

    def __init__(self):
        self.service = ConversationService()

    async def post_message(self, conversation_id: str, message: MessageRequest, db: Session, user_id: str):
        # Store the message in the database with the provided user_id
        stored_message = self.service.store_message(conversation_id, message, db, user_id)
        
        # Continue with AI processing after storing the message
        message_stream = self.service.message_stream(conversation_id)
        return StreamingResponse(message_stream, media_type="text/event-stream")
    
    def create_conversation(self, conversation: CreateConversationRequest) -> CreateConversationResponse:
        conversation_id, message = self.service.create_conversation(conversation)
        
        # Ensure the message is a string
        if isinstance(message, str):
            return CreateConversationResponse(message=message, conversation_id=conversation_id)
        else:
            raise ValueError("Message returned from service is not a string")