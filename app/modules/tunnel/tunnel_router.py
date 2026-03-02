"""
Tunnel router: workspace metadata and socket status (Socket.IO path).
"""

from fastapi import APIRouter, Depends, HTTPException

from app.modules.auth.auth_service import AuthService
from app.modules.tunnel.tunnel_service import get_tunnel_service
from app.modules.tunnel.socket_service import get_socket_service
from app.modules.utils.logger import setup_logger
from pydantic import BaseModel, Field

logger = setup_logger(__name__)
router = APIRouter()


class WorkspaceMetadataResponse(BaseModel):
    """Workspace tunnel metadata (no Cloudflare fields)."""
    workspace_id: str
    repo_url: str = Field(default="", description="Normalised repo URL if known")
    socket_online: bool = Field(description="True if a socket is connected for this workspace")


class WorkspaceSocketStatusResponse(BaseModel):
    """Socket connection status for a workspace."""
    workspace_id: str
    online: bool


def _validate_workspace_id_hex(workspace_id: str) -> None:
    if len(workspace_id) != 16 or not all(c in "0123456789abcdef" for c in workspace_id.lower()):
        raise HTTPException(status_code=400, detail="workspace_id must be 16 hex characters")


@router.get(
    "/tunnel/workspace/{workspace_id}",
    response_model=WorkspaceMetadataResponse,
    description="Get workspace metadata and socket status.",
)
async def get_workspace_tunnel(
    workspace_id: str,
    user=Depends(AuthService.check_auth),
):
    _validate_workspace_id_hex(workspace_id)
    tunnel_service = get_tunnel_service()
    record = await tunnel_service.get_workspace_tunnel_record_async(workspace_id)
    if record and record.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=404, detail="Workspace not found")
    repo_url = str(record["repo_url"]) if record and record.get("repo_url") else ""
    socket_online = get_socket_service().is_workspace_online(workspace_id)
    return WorkspaceMetadataResponse(
        workspace_id=workspace_id,
        repo_url=repo_url,
        socket_online=socket_online,
    )


@router.get(
    "/tunnel/workspace/{workspace_id}/socket-status",
    response_model=WorkspaceSocketStatusResponse,
    description="Check if a socket is connected for this workspace.",
)
async def get_workspace_socket_status(
    workspace_id: str,
    user=Depends(AuthService.check_auth),
):
    _validate_workspace_id_hex(workspace_id)
    tunnel_service = get_tunnel_service()
    record = await tunnel_service.get_workspace_tunnel_record_async(workspace_id)
    if record and record.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=404, detail="Workspace not found")
    online = get_socket_service().is_workspace_online(workspace_id)
    return WorkspaceSocketStatusResponse(workspace_id=workspace_id, online=online)
