from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class UserConversationListRequest(BaseModel):
    user_id: str
    start: int = 0  # Default start index
    limit: int = 10  # Default limit


class UserConversationListResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    status: Optional[str]
    project_ids: Optional[List[str]]
    created_at: str
    updated_at: str


class CreateUser(BaseModel):
    uid: str
    email: str
    display_name: str
    email_verified: bool
    created_at: datetime
    last_login_at: datetime
    provider_info: dict
    provider_username: str
