import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.tunnel.tunnel_service import get_tunnel_service
from app.modules.tunnel.cloudflare_tunnel_service import get_cloudflare_tunnel_service
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


class TunnelRegisterRequest(BaseModel):
    tunnel_url: str = Field(
        ..., description="Public tunnel URL (e.g., https://xyz.trycloudflare.com)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation id to scope tunnel to a conversation",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user id (falls back to auth token user_id if not provided)",
    )
    workspace_id: Optional[str] = Field(
        default=None,
        description="Optional workspace fingerprint/id for multi-workspace support",
    )
    local_port: Optional[int] = Field(
        default=None,
        description="Optional local port being exposed (for debugging/ops)",
    )
    repository: Optional[str] = Field(
        default=None,
        description="Repository identifier (e.g. owner/repo) for workspace-scoped tunnel; one active tunnel per (user, repository, branch)",
    )
    branch: Optional[str] = Field(
        default=None,
        description="Branch name for workspace-scoped tunnel; one active tunnel per (user, repository, branch)",
    )


class TunnelRegisterResponse(BaseModel):
    message: str
    tunnel_url: str
    conversation_id: Optional[str] = None
    prefer_named_tunnel: Optional[bool] = Field(
        default=None,
        description="When True, server has named tunnel configured; extension should prefer POST /tunnels/provision and re-register with that URL.",
    )


class TunnelCapabilitiesResponse(BaseModel):
    named_tunnel_available: bool = Field(
        description="When True, extension should call POST /tunnels/provision first and use that tunnel instead of quick tunnel.",
    )


class TunnelStatusResponse(BaseModel):
    connected: bool
    tunnel_url: Optional[str] = None
    conversation_id: Optional[str] = None


class TunnelProvisionRequest(BaseModel):
    local_port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port the extension will use with cloudflared: --url http://localhost:PORT. Ingress is configured to this port so traffic reaches your local server. If omitted, server uses CLOUDFLARE_TUNNEL_INGRESS_PORT or 3001.",
    )


class TunnelProvisionResponse(BaseModel):
    tunnel_id: str
    tunnel_name: str
    tunnel_token: str
    tunnel_url: str
    ingress_configured: bool = Field(
        default=False,
        description="Whether ingress is configured. If False, extension should use quick tunnel.",
    )


@router.post(
    "/tunnels/provision",
    response_model=TunnelProvisionResponse,
    description="Provision a named Cloudflare tunnel for the user. Returns token for cloudflared.",
)
async def provision_tunnel(
    body: Optional[TunnelProvisionRequest] = None,
    user=Depends(AuthService.check_auth),
    _db: Session = Depends(get_db),
):
    """
    Provision a named Cloudflare tunnel for the authenticated user.

    Named tunnels are more reliable than quick tunnels:
    - Persistent URL (doesn't change on restart)
    - Auto-reconnection support
    - Better uptime (99%+)

    Pass local_port (e.g. 8013) so ingress forwards to the same port you use with:
    cloudflared tunnel run --token <TOKEN> --url http://localhost:<local_port>
    If omitted, port from the user's last tunnel registration (stored in Redis) is used.
    """
    user_id = user["user_id"]
    local_port = body.local_port if body else None
    if local_port is None:
        tunnel_service = get_tunnel_service()
        info = tunnel_service.get_tunnel_info(user_id=user_id)
        if info and isinstance(info.get("local_port"), int):
            local_port = info["local_port"]
            logger.info(
                f"[Tunnel] Using local_port={local_port} from last tunnel registration for user_id={user_id}"
            )
    cf_service = get_cloudflare_tunnel_service()

    if not cf_service.is_configured():
        logger.warning(
            f"[Tunnel] Named tunnels not configured, user {user_id} should use quick tunnel"
        )
        raise HTTPException(
            status_code=503,
            detail="Named tunnels not configured on server. Please use quick tunnel instead.",
        )

    result = await cf_service.provision_tunnel_for_user(user_id, local_port=local_port)
    if not result:
        logger.error(f"[Tunnel] Failed to provision named tunnel for user {user_id}")
        raise HTTPException(
            status_code=500,
            detail="Failed to provision tunnel. Please try again or use quick tunnel.",
        )

    logger.info(
        f"[Tunnel] Provisioned named tunnel for user {user_id}: {result['tunnel_name']}"
    )
    return TunnelProvisionResponse(**result)


@router.get(
    "/tunnels/capabilities",
    response_model=TunnelCapabilitiesResponse,
    description="Check if named tunnel is available. Extension should call this before starting a tunnel and prefer POST /tunnels/provision when true.",
)
async def tunnel_capabilities(
    user=Depends(AuthService.check_auth),
):
    """Return whether the server can provision a named tunnel (so the extension can avoid quick tunnel)."""
    cf_service = get_cloudflare_tunnel_service()
    named_available = cf_service.is_configured() and cf_service.has_domain()
    return TunnelCapabilitiesResponse(named_tunnel_available=named_available)


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
    # Use user_id from request body if provided, otherwise fall back to auth token
    user_id = req.user_id or user["user_id"]
    tunnel_service = get_tunnel_service()

    # Tunnel type is determined by the extension: quick = trycloudflare.com, named = our domain (e.g. potpie.ai)
    tunnel_type = "quick" if "trycloudflare.com" in req.tunnel_url else "named"
    logger.info(
        f"[TunnelRouter] 📝 Registration request: type={tunnel_type}, user_id={user_id}, "
        f"conversation_id={req.conversation_id}, tunnel_url={req.tunnel_url}, "
        f"local_port={req.local_port}, repository={req.repository}, branch={req.branch}"
    )

    # In development, allow http://localhost or http://127.0.0.1 (no cloudflared)
    is_development = (os.getenv("ENV") or "").strip().lower() == "development"
    if is_development:
        allowed = (
            req.tunnel_url.startswith("https://")
            or req.tunnel_url.startswith("http://localhost")
            or req.tunnel_url.startswith("http://127.0.0.1")
        )
    else:
        allowed = req.tunnel_url.startswith("https://")

    if not allowed:
        logger.error(f"[TunnelRouter] ❌ Invalid tunnel_url format: {req.tunnel_url}")
        raise HTTPException(
            status_code=400,
            detail="tunnel_url must start with https:// (or http://localhost / http://127.0.0.1 when ENV=development)",
        )

    ok = tunnel_service.register_tunnel(
        user_id=user_id,
        tunnel_url=req.tunnel_url,
        conversation_id=req.conversation_id,
        local_port=req.local_port,
        repository=req.repository,
        branch=req.branch,
    )
    if not ok:
        logger.error(
            f"[TunnelRouter] ❌ Failed to register tunnel: user_id={user_id}, "
            f"conversation_id={req.conversation_id}, tunnel_url={req.tunnel_url}"
        )
        raise HTTPException(status_code=500, detail="Failed to register tunnel")

    # Verify registration was successful (by workspace when repo+branch provided, else conversation/user)
    verify_url = tunnel_service.get_tunnel_url(
        user_id,
        req.conversation_id,
        repository=req.repository,
        branch=req.branch,
    )
    if verify_url != req.tunnel_url:
        logger.warning(
            f"[TunnelRouter] ⚠️ Registration verification failed: "
            f"expected={req.tunnel_url}, got={verify_url}"
        )
    else:
        logger.info(
            f"[TunnelRouter] ✅ Registration verified: type={tunnel_type}, user_id={user_id}, "
            f"conversation_id={req.conversation_id}, tunnel_url={req.tunnel_url}"
        )

    # Hint extension to use named tunnel when it registered a quick tunnel but server has named tunnel configured
    prefer_named = False
    if tunnel_type == "quick":
        cf_service = get_cloudflare_tunnel_service()
        if cf_service.is_configured() and cf_service.has_domain():
            prefer_named = True
            logger.info(
                f"[TunnelRouter] Server has named tunnel configured; response includes prefer_named_tunnel=true so extension can switch via POST /tunnels/provision"
            )

    return TunnelRegisterResponse(
        message="Tunnel registered",
        tunnel_url=req.tunnel_url,
        conversation_id=req.conversation_id,
        prefer_named_tunnel=prefer_named if prefer_named else None,
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
    ok = tunnel_service.unregister_tunnel(
        user_id=user_id, conversation_id=conversation_id
    )
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
    tunnel_url = tunnel_service.get_tunnel_url(
        user_id=user_id, conversation_id=conversation_id
    )

    logger.info(
        f"[TunnelRouter] Status check: user_id={user_id}, conversation_id={conversation_id}, "
        f"tunnel_url={tunnel_url}, connected={bool(tunnel_url)}"
    )

    return TunnelStatusResponse(
        connected=bool(tunnel_url),
        tunnel_url=tunnel_url,
        conversation_id=conversation_id,
    )
