from .service import ConversationService
from .schema import CreateConversationRequest, CreateConversationResponse

class ConversationController:

    def __init__(self):
        self.service = ConversationService()

    def create_conversation(self, conversation: CreateConversationRequest) -> CreateConversationResponse:
        conversation_id, message = self.service.create_conversation(conversation)
        
        # Ensure the message is a string
        if isinstance(message, str):
            return CreateConversationResponse(message=message, conversation_id=conversation_id)
        else:
            raise ValueError("Message returned from service is not a string")