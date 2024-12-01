from pydantic import BaseModel

class UsageResponse(BaseModel):
    conversation_count: int
    messages_count: int
