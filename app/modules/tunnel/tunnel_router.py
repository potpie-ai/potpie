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
    provisioned_tunnel_url: Optional[str] = Field(
        default=None,
        description="When set, server created a workspace tunnel; extension should use this URL and provisioned_tunnel_token to run cloudflared and then re-register.",
    )
    provisioned_tunnel_token: Optional[str] = Field(
        default=None,
        description="Token for cloudflared when provisioned_tunnel_url is set; use this tunnel instead of the one you registered with.",
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
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation id from provisionContext.")
    repository: Optional[str] = Field(
        default=None,
        description="Workspace repo (e.g. owner/repo). When set, provision is workspace-scoped: one tunnel per repo.",
    )
    branch: Optional[str] = Field(default=None, description="Optional branch for workspace context.")


class TunnelProvisionResponse(BaseModel):
    tunnel_id: str
    tunnel_name: str
    tunnel_token: str
    tunnel_url: str
    ingress_configured: bool = Field(
        default=False,
        description="True = use this tunnel with cloudflared (tunnel_url is reachable). False = extension may fall back to quick tunnel.",
    )
    routing_mode: Optional[str] = Field(
        default=None,
        description="When 'wildcard', tunnel is reachable via tunnel_url (Router proxies). When null/other, legacy ingress.",
    )


# --- Wildcard + workspace tunnel APIs ---

class TunnelHeartbeatRequest(BaseModel):
    user_id: str = Field(..., description="User ID (must match auth)")
    workspace_id: str = Field(..., description="16-char hex workspace_id")
    tunnel_id: str = Field(..., description="Cloudflare tunnel UUID")
    repo_url: str = Field(..., description="Normalised repo URL (e.g. github.com/owner/repo)")
    local_port: int = Field(..., ge=1, le=65535, description="Local server port")
    status: str = Field(default="online", description="online | offline")


class WorkspaceProvisionRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional; defaults to auth user")
    workspace_id: str = Field(..., description="16-char hex workspace_id")
    repo_url: str = Field(..., description="Normalised repo URL")
    local_port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port the extension uses with cloudflared: --url http://localhost:PORT. Backend sets tunnel ingress to this origin so traffic reaches LocalServer. If omitted, server uses CLOUDFLARE_TUNNEL_INGRESS_PORT or 3001.",
    )


class WorkspaceProvisionResponse(BaseModel):
    workspace_id: str
    tunnel_id: str


class WorkspaceMetadataResponse(BaseModel):
    workspace_id: str
    tunnel_id: str
    repo_url: str


class WorkspaceCredentialResponse(BaseModel):
    tunnel_id: str
    credential_json: str = Field(..., description="Tunnel token for cloudflared")


@router.post(
    "/tunnel/heartbeat",
    description="Update workspace presence for wildcard routing. Extension calls every 30s; TTL 90s.",
)
async def tunnel_heartbeat(
    body: TunnelHeartbeatRequest,
    user=Depends(AuthService.check_auth),
):
    if body.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="user_id must match authenticated user")
    if len(body.workspace_id) != 16 or not all(c in "0123456789abcdef" for c in body.workspace_id.lower()):
        raise HTTPException(status_code=400, detail="workspace_id must be 16 hex characters")
    tunnel_service = get_tunnel_service()
    ok = tunnel_service.update_workspace_presence(
        workspace_id=body.workspace_id,
        tunnel_id=body.tunnel_id,
        user_id=body.user_id,
        repo_url=body.repo_url,
        local_port=body.local_port,
        status=body.status,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update workspace presence")
    return {"message": "ok", "status": body.status}


@router.get(
    "/tunnel/workspace/{workspace_id}",
    response_model=WorkspaceMetadataResponse,
    description="Get workspace tunnel metadata. 404 if not provisioned.",
)
async def get_workspace_tunnel(
    workspace_id: str,
    user=Depends(AuthService.check_auth),
):
    tunnel_service = get_tunnel_service()
    record = tunnel_service.get_workspace_tunnel_record(workspace_id)
    if not record or record.get("status") != "active":
        raise HTTPException(status_code=404, detail="Workspace tunnel not provisioned")
    if record.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=404, detail="Workspace tunnel not provisioned")
    return WorkspaceMetadataResponse(
        workspace_id=workspace_id,
        tunnel_id=str(record["tunnel_id"]),
        repo_url=str(record["repo_url"]),
    )


@router.post(
    "/tunnel/workspace/provision",
    response_model=WorkspaceProvisionResponse,
    description="Provision a workspace-scoped Cloudflare tunnel (potpie-ws-{workspace_id}, no DNS/ingress).",
)
async def provision_workspace_tunnel(
    body: WorkspaceProvisionRequest,
    user=Depends(AuthService.check_auth),
):
    from app.modules.tunnel.cloudflare_tunnel_service import get_cloudflare_tunnel_service
    from app.modules.integrations.token_encryption import encrypt_token
    user_id = body.user_id or user["user_id"]
    if len(body.workspace_id) != 16 or not all(c in "0123456789abcdef" for c in body.workspace_id.lower()):
        raise HTTPException(status_code=400, detail="workspace_id must be 16 hex characters")
    tunnel_service = get_tunnel_service()
    existing = tunnel_service.get_workspace_tunnel_record(body.workspace_id)
    if existing and existing.get("status") == "active" and existing.get("user_id") == user_id:
        return WorkspaceProvisionResponse(workspace_id=body.workspace_id, tunnel_id=existing["tunnel_id"])
    cf = get_cloudflare_tunnel_service()
    result = await cf.provision_workspace_tunnel(body.workspace_id, local_port=body.local_port)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to provision workspace tunnel")
    encrypted = encrypt_token(result["tunnel_token"])
    ok = tunnel_service.set_workspace_tunnel_record(
        workspace_id=body.workspace_id,
        user_id=user_id,
        repo_url=body.repo_url,
        tunnel_id=result["tunnel_id"],
        tunnel_name=result["tunnel_name"],
        tunnel_credential_encrypted=encrypted,
        status="active",
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to store workspace tunnel record")
    return WorkspaceProvisionResponse(workspace_id=body.workspace_id, tunnel_id=result["tunnel_id"])


@router.get(
    "/tunnel/credential/{workspace_id}",
    response_model=WorkspaceCredentialResponse,
    description="Get tunnel credential for workspace. 404 if not provisioned or not owned by user.",
)
async def get_workspace_credential(
    workspace_id: str,
    user=Depends(AuthService.check_auth),
):
    from app.modules.integrations.token_encryption import decrypt_token
    tunnel_service = get_tunnel_service()
    record = tunnel_service.get_workspace_tunnel_record(workspace_id)
    if not record or record.get("status") != "active":
        raise HTTPException(status_code=404, detail="Workspace tunnel not provisioned")
    if record.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=404, detail="Workspace tunnel not provisioned")
    try:
        credential = decrypt_token(str(record["tunnel_credential_encrypted"]))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to decrypt credential")
    return WorkspaceCredentialResponse(tunnel_id=str(record["tunnel_id"]), credential_json=credential)


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

    When repository is provided, provision is workspace-scoped: one tunnel per workspace
    (potpie-ws-{workspace_id}), so different repos get different tunnels and no stale-tunnel mix-up.
    """
    user_id = user["user_id"]
    body = body or TunnelProvisionRequest()
    local_port = body.local_port
    repository = body.repository
    branch = body.branch

    # Workspace-scoped provision when repository is sent (one tunnel per repo)
    if repository:
        from app.modules.tunnel.tunnel_service import (
            normalise_repo_url,
            compute_workspace_id,
            _tunnel_wildcard_enabled,
            _tunnel_wildcard_domain,
        )
        from app.modules.integrations.token_encryption import encrypt_token

        tunnel_service = get_tunnel_service()
        repo_normalised = normalise_repo_url(repository)
        if not repo_normalised:
            raise HTTPException(
                status_code=400,
                detail="repository could not be normalised (e.g. provide owner/repo or full repo URL).",
            )
        workspace_id = compute_workspace_id(user_id, repo_normalised)
        existing = tunnel_service.get_workspace_tunnel_record(workspace_id)
        if existing and existing.get("status") == "active" and existing.get("user_id") == user_id:
            # Return existing workspace tunnel; extension needs credential from GET /tunnel/credential/{workspace_id}
            from app.modules.integrations.token_encryption import decrypt_token
            try:
                token = decrypt_token(str(existing["tunnel_credential_encrypted"]))
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to decrypt workspace tunnel credential.")
            tunnel_url = (
                f"https://{workspace_id}.{_tunnel_wildcard_domain()}"
                if _tunnel_wildcard_enabled() and _tunnel_wildcard_domain()
                else f"https://{existing['tunnel_id']}.cfargotunnel.com"
            )
            logger.info(
                f"[Tunnel] Returning existing workspace tunnel for workspace_id={workspace_id}, repo={repository}"
            )
            return TunnelProvisionResponse(
                tunnel_id=existing["tunnel_id"],
                tunnel_name=existing["tunnel_name"],
                tunnel_token=token,
                tunnel_url=tunnel_url,
                ingress_configured=True,
                routing_mode="wildcard" if (_tunnel_wildcard_enabled() and _tunnel_wildcard_domain()) else None,
            )
        cf_service = get_cloudflare_tunnel_service()
        if not cf_service.is_configured():
            raise HTTPException(
                status_code=503,
                detail="Named tunnels not configured on server. Please use quick tunnel instead.",
            )
        result = await cf_service.provision_workspace_tunnel(workspace_id, local_port=body.local_port)
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to provision workspace tunnel. Please try again.",
            )
        encrypted = encrypt_token(result["tunnel_token"])
        ok = tunnel_service.set_workspace_tunnel_record(
            workspace_id=workspace_id,
            user_id=user_id,
            repo_url=repo_normalised,
            tunnel_id=result["tunnel_id"],
            tunnel_name=result["tunnel_name"],
            tunnel_credential_encrypted=encrypted,
            status="active",
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to store workspace tunnel record.")
        tunnel_url = (
            f"https://{workspace_id}.{_tunnel_wildcard_domain()}"
            if _tunnel_wildcard_enabled() and _tunnel_wildcard_domain()
            else f"https://{result['tunnel_id']}.cfargotunnel.com"
        )
        logger.info(
            f"[Tunnel] Provisioned workspace tunnel for workspace_id={workspace_id}, repo={repository}: {result['tunnel_name']}"
        )
        return TunnelProvisionResponse(
            tunnel_id=result["tunnel_id"],
            tunnel_name=result["tunnel_name"],
            tunnel_token=result["tunnel_token"],
            tunnel_url=tunnel_url,
            ingress_configured=True,
            routing_mode="wildcard" if (_tunnel_wildcard_enabled() and _tunnel_wildcard_domain()) else None,
        )

    # Workspace-only: repository is required (no user-level provision)
    raise HTTPException(
        status_code=400,
        detail="repository is required for tunnel provision (workspace-level only). Send repository (e.g. owner/repo) in the request body.",
    )


@router.get(
    "/tunnels/capabilities",
    response_model=TunnelCapabilitiesResponse,
    description="Check if named tunnel is available. Extension should call this before starting a tunnel and prefer POST /tunnels/provision when true.",
)
async def tunnel_capabilities(
    user=Depends(AuthService.check_auth),
):
    """Return whether the server can provision a named tunnel (so the extension can avoid quick tunnel). Workspace tunnels work with or without CLOUDFLARE_TUNNEL_DOMAIN (cfargotunnel.com or wildcard)."""
    cf_service = get_cloudflare_tunnel_service()
    named_available = cf_service.is_configured()
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

    # For workspace (repository+branch): ensure we have a workspace tunnel and use its URL for register.
    # If none exists, provision one so every workspace gets a tunnel and we can proxy correctly.
    effective_tunnel_url = req.tunnel_url
    provisioned_tunnel_url: Optional[str] = None
    provisioned_tunnel_token: Optional[str] = None

    if req.repository and req.branch:
        from app.modules.tunnel.tunnel_service import (
            normalise_repo_url,
            compute_workspace_id,
            _tunnel_wildcard_enabled,
            _tunnel_wildcard_domain,
        )
        from app.modules.integrations.token_encryption import encrypt_token

        repo_normalised = normalise_repo_url(req.repository)
        if repo_normalised:
            workspace_id = compute_workspace_id(user_id, repo_normalised)
            existing = tunnel_service.get_workspace_tunnel_record(workspace_id)
            if not existing or existing.get("status") != "active" or existing.get("user_id") != user_id:
                # No workspace tunnel for this workspace: provision one (POST tunnel) so we use it.
                cf_service = get_cloudflare_tunnel_service()
                if cf_service.is_configured():
                    result = await cf_service.provision_workspace_tunnel(
                        workspace_id, local_port=req.local_port
                    )
                    if result:
                        encrypted = encrypt_token(result["tunnel_token"])
                        ok_store = tunnel_service.set_workspace_tunnel_record(
                            workspace_id=workspace_id,
                            user_id=user_id,
                            repo_url=repo_normalised,
                            tunnel_id=result["tunnel_id"],
                            tunnel_name=result["tunnel_name"],
                            tunnel_credential_encrypted=encrypted,
                            status="active",
                        )
                        if ok_store:
                            effective_tunnel_url = (
                                f"https://{workspace_id}.{_tunnel_wildcard_domain()}"
                                if _tunnel_wildcard_enabled() and _tunnel_wildcard_domain()
                                else f"https://{result['tunnel_id']}.cfargotunnel.com"
                            )
                            provisioned_tunnel_url = effective_tunnel_url
                            provisioned_tunnel_token = result["tunnel_token"]
                            logger.info(
                                f"[TunnelRouter] Provisioned workspace tunnel on register: workspace_id={workspace_id}, "
                                f"tunnel_url={effective_tunnel_url}; extension should use this URL and token."
                            )
            else:
                # Use our stored workspace tunnel URL so register and presence match router expectations.
                effective_tunnel_url = (
                    f"https://{workspace_id}.{_tunnel_wildcard_domain()}"
                    if _tunnel_wildcard_enabled() and _tunnel_wildcard_domain()
                    else f"https://{existing['tunnel_id']}.cfargotunnel.com"
                )
                if effective_tunnel_url != req.tunnel_url:
                    logger.info(
                        f"[TunnelRouter] Using workspace tunnel URL for register: {effective_tunnel_url} (request had {req.tunnel_url})"
                    )
                # Re-apply ingress config on every register so the hostname is always
                # set in the tunnel config (covers cases where provision ran before the
                # hostname-route code was added, or the config was cleared).
                cf_service = get_cloudflare_tunnel_service()
                if cf_service.is_configured():
                    tunnel_id = existing.get("tunnel_id")
                    if tunnel_id:
                        logger.info(
                            f"[TunnelRouter] Refreshing tunnel ingress for existing workspace tunnel: "
                            f"workspace_id={workspace_id}, tunnel_id={tunnel_id}"
                        )
                        await cf_service.ensure_tunnel_ingress(
                            tunnel_id=tunnel_id,
                            workspace_id=workspace_id,
                            local_port=req.local_port,
                        )

    ok = tunnel_service.register_tunnel(
        user_id=user_id,
        tunnel_url=effective_tunnel_url,
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
    if verify_url != effective_tunnel_url:
        logger.warning(
            f"[TunnelRouter] ⚠️ Registration verification failed: "
            f"expected={effective_tunnel_url}, got={verify_url}"
        )
    else:
        logger.info(
            f"[TunnelRouter] ✅ Registration verified: type={tunnel_type}, user_id={user_id}, "
            f"conversation_id={req.conversation_id}, tunnel_url={effective_tunnel_url}"
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
        tunnel_url=effective_tunnel_url,
        conversation_id=req.conversation_id,
        prefer_named_tunnel=prefer_named if prefer_named else None,
        provisioned_tunnel_url=provisioned_tunnel_url,
        provisioned_tunnel_token=provisioned_tunnel_token,
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
