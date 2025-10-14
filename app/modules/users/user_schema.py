from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel


class UserConversationListRequest(BaseModel):
    user_id: str
    start: int = 0  # Default start index
    limit: int = 10  # Default limit


class UserConversationListResponse(BaseModel):
    id: str
    title: Optional[str]
    status: Optional[str]
    project_ids: Optional[List[str]]
    agent_id: Optional[str]
    repository: Optional[str]
    branch: Optional[str]
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
    provider_username: Optional[str] = None  # Optional for email/password users


class TokenMetadata(BaseModel):
    """Metadata about how the token was created"""

    created_via: str
    provider: str
    auth_method: str
    created_at: Optional[datetime] = None


class ProviderInfo(BaseModel):
    """Structured format for provider_info JSONB field"""

    access_token: Optional[str] = None  # Optional for email/password users
    providerId: str
    uid: str
    displayName: Optional[str] = None
    email: Optional[str] = None
    token_type: Optional[Literal["oauth", "app_user", "pat"]] = None
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    installation_id: Optional[int] = None
    token_metadata: Optional[TokenMetadata] = None

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat() if v else None},
        "extra": "allow",
    }


class UserProfileResponse(BaseModel):
    user_id: str
    profile_pic_url: Optional[str] = None
