"""
Workspace tunnel metadata (Socket.IO path).

Data is stored in Redis only (see tunnel_service.get/set_workspace_tunnel_record).
One record per workspace (workspace_id = sha256(user_id:repo_url)[:16]).
Used for workspace metadata (e.g. repo_url) and ownership; connectivity is via Socket.IO.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WorkspaceTunnelRecord(BaseModel):
    """
    Schema for workspace record stored in Redis (not in DB).
    Redis key: workspace_tunnel:{workspace_id}. Value has no workspace_id; add it when building from key.
    """

    workspace_id: Optional[str] = Field(default=None, description="16-char hex; in Redis key, not in value")
    user_id: str = Field(..., description="Owner user id")
    repo_url: str = Field(..., description="Normalised repo URL")
    status: str = Field(default="active", description="active | deprovisioned")
    provisioned_at: Optional[datetime] = Field(default=None, description="Not stored in Redis; optional")

    class Config:
        extra = "ignore"
