from pydantic import BaseModel, EmailStr
from typing import Optional, List


class ShareChatRequest(BaseModel):
    conversation_id: str  
    recipientEmails: Optional[List[EmailStr]]=None
    share_with_link: bool = False


class ShareChatResponse(BaseModel):
    message: str  
    sharedID: str  


class SharedChatResponse(BaseModel):
    chat: dict 