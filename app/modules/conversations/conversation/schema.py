from pydantic import BaseModel
from typing import List
from datetime import datetime
from .model import ConversationStatus

class CreateConversation(BaseModel):
    user_id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    agent_ids: List[str]


class CreateConversation(BaseModel):
    user_id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    agent_ids: List[str]

class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    agent_ids: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ConversationInfoResponse(BaseModel):
    id: str
    agent_ids: List[str]
    project_ids: List[str]

    class Config:
        orm_mode = True
