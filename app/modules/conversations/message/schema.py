from pydantic import BaseModel
from typing import  Optional
from datetime import datetime
from ..message.model import MessageType

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    content: str
    sender_id: Optional[str] = None
    type: str
    created_at: datetime

    class Config:
        orm_mode = True