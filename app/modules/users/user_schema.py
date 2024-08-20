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
