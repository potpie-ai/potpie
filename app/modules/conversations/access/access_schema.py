from typing import List

from pydantic import BaseModel, EmailStr


class ShareChatRequest(BaseModel):
    conversation_id: str
    recipientEmails: List[EmailStr]


class ShareChatResponse(BaseModel):
    message: str
    sharedID: str


class SharedChatResponse(BaseModel):
    chat: dict
