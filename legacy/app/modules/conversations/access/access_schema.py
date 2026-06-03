from typing import List, Optional

from pydantic import BaseModel, EmailStr

from app.modules.conversations.conversation.conversation_model import Visibility


class ShareChatRequest(BaseModel):
    conversation_id: str
    recipientEmails: Optional[List[EmailStr]] = None
    visibility: Visibility


class ShareChatResponse(BaseModel):
    message: str
    sharedID: str


class SharedChatResponse(BaseModel):
    chat: dict


class RemoveAccessRequest(BaseModel):
    emails: List[EmailStr]
