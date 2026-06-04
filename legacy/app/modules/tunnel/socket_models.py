"""
Pydantic models for Socket.IO workspace tunnel events.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class AuthPayload(BaseModel):
    """Payload for auth event (client -> server). First message after connect."""

    token: str = Field(..., description="Firebase ID token (JWT)")


class AuthSuccessPayload(BaseModel):
    """Payload for auth_success event (server -> client)."""

    uid: str = Field(..., description="Authenticated user id (Firebase uid)")


class WorkspaceRegisterPayload(BaseModel):
    """Payload for register_workspace event (client -> server)."""

    workspace_id: str = Field(..., description="16-char hex workspace_id")
    repo_url: str = Field(..., description="Normalised repo URL")
    user_id: str = Field(..., description="Owner user id")


class ToolCallEvent(BaseModel):
    """Payload for tool_call event (server -> client)."""

    correlation_id: str = Field(..., description="Unique ID to match response")
    endpoint: str = Field(..., description="API path e.g. /api/files/read-batch")
    payload: dict = Field(default_factory=dict, description="Request body")
    timeout: Optional[float] = Field(default=None, description="Timeout in seconds")


class ToolResponseEvent(BaseModel):
    """Payload for tool_response event (client -> server)."""

    correlation_id: str = Field(..., description="Must match tool_call correlation_id")
    success: bool = Field(..., description="Whether the operation succeeded")
    result: Optional[Any] = Field(default=None, description="Response body on success")
    error: Optional[str] = Field(default=None, description="Error message on failure")


class WorkspaceHeartbeatPayload(BaseModel):
    """Payload for heartbeat event (client -> server)."""

    workspace_id: str = Field(..., description="16-char hex workspace_id")
