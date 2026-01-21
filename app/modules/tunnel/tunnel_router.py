from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.tunnel.tunnel_service import get_tunnel_service
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


class TunnelRegisterRequest(BaseModel):
    tunnel_url: str = Field(..., description="Public tunnel URL (e.g., https://xyz.trycloudflare.com)")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation id to scope tunnel to a conversation",
    )
    workspace_id: Optional[str] = Field(
        default=None,
        description="Optional workspace fingerprint/id for multi-workspace support",
    )
    local_port: Optional[int] = Field(
        default=None,
        description="Optional local port being exposed (for debugging/ops)",
    )


class TunnelRegisterResponse(BaseModel):
    message: str
    tunnel_url: str
    conversation_id: Optional[str] = None


class TunnelStatusResponse(BaseModel):
    connected: bool
    tunnel_url: Optional[str] = None
    conversation_id: Optional[str] = None


@router.post(
    "/tunnels/register",
    response_model=TunnelRegisterResponse,
    description="Register a local tunnel URL for the authenticated user.",
)
async def register_tunnel(
    req: TunnelRegisterRequest,
    user=Depends(AuthService.check_auth),
    _db: Session = Depends(get_db),
):
    user_id = user["user_id"]
    tunnel_service = get_tunnel_service()

    if not req.tunnel_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="tunnel_url must start with https://")

    ok = tunnel_service.register_tunnel(
        user_id=user_id, tunnel_url=req.tunnel_url, conversation_id=req.conversation_id
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to register tunnel")

    logger.info(
        f"Registered tunnel for user_id={user_id} conversation_id={req.conversation_id}: {req.tunnel_url}"
    )
    return TunnelRegisterResponse(
        message="Tunnel registered",
        tunnel_url=req.tunnel_url,
        conversation_id=req.conversation_id,
    )


@router.post(
    "/tunnels/unregister",
    description="Unregister the local tunnel URL for the authenticated user.",
)
async def unregister_tunnel(
    conversation_id: Optional[str] = None,
    user=Depends(AuthService.check_auth),
    _db: Session = Depends(get_db),
):
    user_id = user["user_id"]
    tunnel_service = get_tunnel_service()
    ok = tunnel_service.unregister_tunnel(user_id=user_id, conversation_id=conversation_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to unregister tunnel")
    return {"message": "Tunnel unregistered", "conversation_id": conversation_id}


@router.get(
    "/tunnels/status",
    response_model=TunnelStatusResponse,
    description="Get tunnel status for the authenticated user.",
)
async def tunnel_status(
    conversation_id: Optional[str] = None,
    user=Depends(AuthService.check_auth),
    _db: Session = Depends(get_db),
):
    user_id = user["user_id"]
    tunnel_service = get_tunnel_service()
    tunnel_url = tunnel_service.get_tunnel_url(user_id=user_id, conversation_id=conversation_id)
    return TunnelStatusResponse(
        connected=bool(tunnel_url),
        tunnel_url=tunnel_url,
        conversation_id=conversation_id,
    )

