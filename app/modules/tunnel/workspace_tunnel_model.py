"""
WorkspaceTunnel schema for workspace-scoped Cloudflare tunnels.

Data is stored in Redis only (see tunnel_service.get/set_workspace_tunnel_record).
One record per workspace (workspace_id = sha256(user_id:repo_url)[:16]).
Tunnel name: potpie-ws-{workspace_id}. No DNS/ingress per workspace;
Router Service proxies to https://{tunnel_id}.cfargotunnel.com.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WorkspaceTunnelRecord(BaseModel):
    """
    Schema for workspace tunnel record stored in Redis (not in DB).
    Redis key: workspace_tunnel:{workspace_id}. Value has no workspace_id; add it when building from key.
    """

    workspace_id: Optional[str] = Field(default=None, description="16-char hex; in Redis key, not in value")
    user_id: str = Field(..., description="Owner user id")
    repo_url: str = Field(..., description="Normalised repo URL")
    tunnel_id: str = Field(..., description="Cloudflare tunnel UUID")
    tunnel_name: str = Field(..., description="e.g. potpie-ws-{workspace_id}")
    tunnel_credential_encrypted: str = Field(..., description="Encrypted tunnel token for cloudflared")
    status: str = Field(default="active", description="active | deprovisioned")
    provisioned_at: Optional[datetime] = Field(default=None, description="Not stored in Redis; optional")

    class Config:
        extra = "ignore"
