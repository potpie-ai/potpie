from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.config import Config
from starlette.responses import RedirectResponse
from typing import Dict, Any, Optional
import hmac
import hashlib
import base64
import json
import time
from app.modules.utils.logger import setup_logger
from app.modules.integrations import hash_user_id

import urllib.parse
import jwt

from app.core.database import get_db
from app.api.router import get_api_key_user
from app.modules.auth.auth_service import AuthService

from .sentry_oauth_v2 import SentryOAuthV2
from .linear_oauth import LinearOAuth
from .jira_oauth import JiraOAuth
from .confluence_oauth import ConfluenceOAuth
from .integrations_service import IntegrationsService
from .integrations_schema import (
    OAuthInitiateRequest,
    OAuthStatusResponse,
    SentryIntegrationStatus,
    SentrySaveRequest,
    SentrySaveResponse,
    LinearIntegrationStatus,
    LinearSaveRequest,
    LinearSaveResponse,
    JiraIntegrationStatus,
    JiraSaveRequest,
    JiraSaveResponse,
    ConfluenceIntegrationStatus,
    ConfluenceSaveRequest,
    ConfluenceSaveResponse,
    IntegrationCreateRequest,
    IntegrationUpdateRequest,
    IntegrationResponse,
    IntegrationListResponse,
    IntegrationType,
    IntegrationStatus,
    IntegrationSaveRequest,
    IntegrationSaveResponse,
)


logger = setup_logger(__name__)
router = APIRouter(prefix="/integrations", tags=["integrations"])


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Remove or redact sensitive header keys"""
    sensitive_keys = {"authorization", "cookie", "set-cookie", "signature", "token"}
    sanitized = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    return sanitized


def truncate_content(content: str, max_length: int = 200) -> str:
    """Truncate content to max_length with ellipsis"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def get_params_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of params with keys and truncated values"""
    return {
        "keys": list(params.keys()),
        "count": len(params),
        "preview": {k: truncate_content(str(v)) for k, v in list(params.items())[:5]},
    }


def get_body_summary(body_text: str) -> Dict[str, Any]:
    """Get a summary of body with length and truncated preview"""
    return {"length": len(body_text), "preview": truncate_content(body_text)}


def get_sentry_oauth() -> SentryOAuthV2:
    """Dependency to get Sentry OAuth integration instance"""
    config = Config()
    return SentryOAuthV2(config)


def get_linear_oauth() -> LinearOAuth:
    """Dependency to get Linear OAuth integration instance"""
    config = Config()
    return LinearOAuth(config)


def get_jira_oauth() -> JiraOAuth:
    """Dependency to get Jira OAuth integration instance"""
    config = Config()
    return JiraOAuth(config)


def get_confluence_oauth() -> ConfluenceOAuth:
    """Dependency to get Confluence OAuth integration instance"""
    config = Config()
    return ConfluenceOAuth(config)


def get_integrations_service(db: Session = Depends(get_db)) -> IntegrationsService:
    """Dependency to get integrations service instance with database session"""
    return IntegrationsService(db)


def _sign_oauth_state(raw_state: str | None, expires: int = 600) -> str | None:
    """Sign a state string with HMAC to protect against tampering.

    Returns a token of the form: base64(payload).hex(hmac)
    payload is JSON: {"u": <raw_state>, "e": <expiry_ts>} encoded in utf-8 then base64-url-safe.
    If raw_state is None or empty, returns None.
    """
    if not raw_state:
        return None

    config = Config()
    secret = config("OAUTH_STATE_SECRET", default="")
    if not secret:
        # If no secret configured, avoid signing (dev fallback)
        return raw_state

    expiry = int(time.time()) + int(expires)
    payload = {"u": raw_state, "e": expiry}
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8")
    sig = hmac.new(
        secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{payload_b64}.{sig}"


def _verify_oauth_state(token: str | None) -> Optional[str]:
    """Verify signed state token and return the embedded user_id (raw_state).

    Returns None if verification fails. If no secret configured, returns token unchanged.
    """
    if not token:
        return None

    config = Config()
    secret = config("OAUTH_STATE_SECRET", default="")
    if not secret:
        # No secret configured – assume token is raw state
        return token

    try:
        if "." not in token:
            return None
        payload_b64, sig = token.rsplit(".", 1)
        expected = hmac.new(
            secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode(
            "utf-8"
        )
        payload = json.loads(payload_json)
        expiry = int(payload.get("e", 0))
        if int(time.time()) > expiry:
            return None
        return payload.get("u")
    except Exception:
        return None


@router.post("/sentry/initiate")
async def initiate_sentry_oauth(
    request: OAuthInitiateRequest,
    sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth),
) -> Dict[str, Any]:
    """Initiate Sentry OAuth flow"""
    try:
        # Log the OAuth initiation details
        logger.info("=== OAuth Initiation Debug ===")
        logger.info(
            "OAuth parameters", redirect_uri=request.redirect_uri, state=request.state
        )

        # Generate authorization URL. Sign the state to prevent tampering.
        signed_state = (
            _sign_oauth_state(request.state)
            if getattr(request, "state", None)
            else None
        )
        auth_url = sentry_oauth.get_authorization_url(
            redirect_uri=request.redirect_uri, state=signed_state
        )

        logger.info("Generated authorization URL for Sentry OAuth")

        return {
            "status": "success",
            "authorization_url": auth_url,
            "message": "OAuth flow initiated successfully",
            "debug_info": {
                "redirect_uri": request.redirect_uri,
                "state": request.state,
                "auth_url": auth_url,
            },
        }
    except Exception as e:
        logger.exception("OAuth initiation failed")
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate OAuth flow: {str(e)}"
        )


@router.get("/sentry/callback")
async def sentry_oauth_callback(
    request: Request,
    sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth),
) -> OAuthStatusResponse:
    """Handle Sentry OAuth callback. Uses signed state to identify user."""
    try:
        # Extract and verify state to get user_id
        state = request.query_params.get("state")
        user_id = _verify_oauth_state(state)
        if not user_id:
            raise HTTPException(
                status_code=400, detail="Invalid or missing OAuth state"
            )

        result = sentry_oauth.handle_callback(request, user_id)
        return OAuthStatusResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {str(e)}")


@router.get("/sentry/status/{user_id}")
async def get_sentry_status(
    user_id: str,
    sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth),
    user: dict = Depends(AuthService.check_auth),
) -> SentryIntegrationStatus:
    """Get Sentry integration status for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot access other users' integration status"
        )

    user_info = sentry_oauth.get_user_info(user_id)

    if not user_info:
        return SentryIntegrationStatus(user_id=user_id, is_connected=False)

    return SentryIntegrationStatus(
        user_id=user_id,
        is_connected=True,
        scope=user_info.get("scope"),
        expires_at=user_info.get("expires_at"),
    )


@router.delete("/sentry/revoke/{user_id}")
async def revoke_sentry_access(
    user_id: str,
    sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Revoke Sentry OAuth access for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot revoke other users' integrations"
        )

    success = sentry_oauth.revoke_access(user_id)

    if success:
        return {
            "status": "success",
            "message": "Sentry access revoked successfully",
            "user_id": user_id,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to revoke Sentry access")


@router.get("/sentry/authorize")
async def sentry_authorize(
    request: Request, sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth)
) -> RedirectResponse:
    """Direct authorization endpoint that redirects to Sentry OAuth"""
    try:
        # Get redirect URI from query params or use default
        redirect_uri = request.query_params.get(
            "redirect_uri", "http://localhost:3000/integrations/sentry/callback"
        )

        # Generate authorization URL
        auth_url = sentry_oauth.get_authorization_url(redirect_uri=redirect_uri)

        # Redirect to Sentry OAuth
        return RedirectResponse(url=auth_url)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to redirect to OAuth: {str(e)}"
        )


@router.get("/sentry/redirect")
async def sentry_redirect_webhook(request: Request) -> Dict[str, Any]:
    """Handle Sentry webhook redirect requests"""
    try:
        # Log the incoming request details
        logger.info("Sentry webhook redirect received")
        logger.info("Request details", method=request.method, url=str(request.url))
        sanitized_headers = sanitize_headers(dict(request.headers))
        logger.debug("Request headers", headers=sanitized_headers)

        # Get query parameters
        query_params = dict(request.query_params)
        params_summary = get_params_summary(query_params)
        logger.info(
            "Query parameters",
            params_keys=params_summary["keys"],
            params_count=params_summary["count"],
        )
        logger.debug("Query parameters", params_summary=params_summary)

        # Try to get request body if it exists
        try:
            body = await request.body()
            if body:
                body_text = (
                    body.decode("utf-8") if isinstance(body, bytes) else str(body)
                )
                body_summary = get_body_summary(body_text)
                logger.info("Request body received", body_length=body_summary["length"])
                logger.debug(
                    "Request body preview", body_preview=body_summary["preview"]
                )
            else:
                logger.info("Request body: (empty)")
        except Exception as e:
            logger.warning("Could not read request body", error=str(e))

        # Log form data if present
        try:
            form_data = await request.form()
            if form_data:
                form_dict = {k: str(v) for k, v in form_data.items()}
                form_summary = get_params_summary(form_dict)
                logger.info(
                    "Form data received",
                    form_fields=form_summary["keys"],
                    form_count=form_summary["count"],
                )
                logger.debug("Form data summary", form_summary=form_summary)
        except Exception as e:
            logger.debug("No form data present", error=str(e))

        return {
            "status": "success",
            "message": "Sentry webhook redirect logged successfully",
            "logged_at": time.time(),
        }

    except Exception as e:
        logger.exception("Error processing Sentry webhook redirect")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Sentry webhook redirect: {str(e)}",
        )


@router.get("/linear/redirect")
async def linear_oauth_redirect(
    request: Request, linear_oauth: LinearOAuth = Depends(get_linear_oauth)
) -> RedirectResponse:
    """Direct authorization endpoint that redirects to Linear OAuth"""
    try:
        # Check if this is actually a callback request (has code parameter)
        if request.query_params.get("code"):
            raise HTTPException(
                status_code=400,
                detail="This endpoint is for OAuth initiation, not callback. Linear should redirect to /api/v1/integrations/linear/callback with the authorization code.",
            )
        # Get redirect URI from query params or construct from current request
        redirect_uri = request.query_params.get(
            "redirect_uri",
            f"https://{request.url.hostname}/api/v1/integrations/linear/callback",
        )

        # Get state parameter if provided
        state = request.query_params.get("state")
        signed_state = _sign_oauth_state(state) if state else None

        # Get scope parameter if provided (default to read)
        scope = request.query_params.get("scope", "read")

        # Generate authorization URL
        auth_url = linear_oauth.get_authorization_url(
            redirect_uri=redirect_uri, state=signed_state, scope=scope
        )

        # Redirect to Linear OAuth
        return RedirectResponse(url=auth_url)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to redirect to Linear OAuth: {str(e)}"
        )


@router.get("/linear/callback")
async def linear_oauth_callback(
    request: Request,
    linear_oauth: LinearOAuth = Depends(get_linear_oauth),
):
    """Handle Linear OAuth callback"""
    try:
        # Extract OAuth parameters
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        # Verify signed state to get user_id
        user_id = _verify_oauth_state(state)
        if not user_id:
            raise HTTPException(
                status_code=400, detail="Invalid or missing OAuth state"
            )

        # If we have an authorization code, exchange it for tokens and save to database
        if code:
            try:
                # Get the integrations service to save the integration
                from app.core.database import SessionLocal

                db = SessionLocal()
                integrations_service = IntegrationsService(db)

                # Create a LinearSaveRequest with the authorization code
                from .integrations_schema import LinearSaveRequest
                from datetime import datetime

                # Always use HTTPS in production
                save_request = LinearSaveRequest(
                    code=code,
                    redirect_uri=f"https://{request.url.hostname}/api/v1/integrations/linear/callback",
                    instance_name="Linear Integration",  # Will be updated with org name in service
                    integration_type="linear",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )

                # Save the integration (this will exchange code for tokens)
                save_result = await integrations_service.save_linear_integration(
                    save_request,
                    user_id,
                )

                db.close()

                # Redirect to frontend with success
                config = Config()
                frontend_url = config("FRONTEND_URL", default="http://localhost:3000")

                # Ensure frontend_url has protocol
                if frontend_url and not frontend_url.startswith(
                    ("http://", "https://")
                ):
                    frontend_url = f"https://{frontend_url}"

                redirect_url = f"{frontend_url}/integrations/linear/redirect?success=true&integration_id={save_result.get('integration_id')}&user_name={save_result.get('user_name', '')}"

                return RedirectResponse(url=redirect_url)

            except Exception as e:
                logger.exception("Failed to save Linear integration")

                # Redirect to frontend with error
                config = Config()
                frontend_url = config("FRONTEND_URL", default="http://localhost:3000")

                # Ensure frontend_url has protocol
                if frontend_url and not frontend_url.startswith(
                    ("http://", "https://")
                ):
                    frontend_url = f"https://{frontend_url}"

                error_message = urllib.parse.quote(str(e), safe="")
                redirect_url = f"{frontend_url}/integrations/linear/redirect?error={error_message}&user_id={user_id}"

                return RedirectResponse(url=redirect_url)
        else:
            # No code provided, redirect to frontend with error
            config = Config()
            frontend_url = config("FRONTEND_URL", default="http://localhost:3000")

            # Ensure frontend_url has protocol
            if frontend_url and not frontend_url.startswith(("http://", "https://")):
                frontend_url = f"https://{frontend_url}"

            error_msg = "No authorization code received from Linear"
            if error:
                error_msg = f"OAuth error: {error}"
            elif error_description:
                error_msg = f"OAuth error: {error_description}"

            error_message = urllib.parse.quote(error_msg, safe="")
            redirect_url = f"{frontend_url}/integrations/linear/redirect?error={error_message}&user_id={user_id}"

            return RedirectResponse(url=redirect_url)
    except Exception as e:
        logger.exception("Linear OAuth callback failed")

        # Redirect to frontend with error
        config = Config()
        frontend_url = config("FRONTEND_URL", default="http://localhost:3000")

        # Ensure frontend_url has protocol
        if frontend_url and not frontend_url.startswith(("http://", "https://")):
            frontend_url = f"https://{frontend_url}"

        error_message = urllib.parse.quote(
            f"Linear OAuth callback failed: {str(e)}", safe=""
        )
        redirect_url = (
            f"{frontend_url}/integrations/linear/redirect?error={error_message}"
        )

        return RedirectResponse(url=redirect_url)


@router.get("/linear/status/{user_id}")
async def get_linear_status(
    user_id: str,
    linear_oauth: LinearOAuth = Depends(get_linear_oauth),
    user: dict = Depends(AuthService.check_auth),
) -> LinearIntegrationStatus:
    """Get Linear integration status for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot access other users' integration status"
        )

    user_info = linear_oauth.get_user_info(user_id)

    if not user_info:
        return LinearIntegrationStatus(user_id=user_id, is_connected=False)

    return LinearIntegrationStatus(
        user_id=user_id,
        is_connected=True,
        scope=user_info.get("scope"),
        expires_at=user_info.get("expires_at"),
    )


@router.delete("/linear/revoke/{user_id}")
async def revoke_linear_access(
    user_id: str,
    linear_oauth: LinearOAuth = Depends(get_linear_oauth),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Revoke Linear OAuth access for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot revoke other users' integrations"
        )

    success = linear_oauth.revoke_access(user_id)

    if success:
        return {
            "status": "success",
            "message": "Linear access revoked successfully",
            "user_id": user_id,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to revoke Linear access")


@router.post("/linear/save")
async def save_linear_integration(
    request: LinearSaveRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> LinearSaveResponse:
    """Save Linear integration after OAuth callback"""
    try:
        # Get authenticated user_id
        user_id = user["user_id"]
        result = await integrations_service.save_linear_integration(request, user_id)
        return LinearSaveResponse(success=True, data=result, error=None)
    except Exception as e:
        logger.exception("Error saving Linear integration")
        return LinearSaveResponse(
            success=False,
            data=None,
            error=f"Failed to save Linear integration: {str(e)}",
        )


@router.post("/jira/initiate")
async def initiate_jira_oauth(
    request: OAuthInitiateRequest,
    jira_oauth: JiraOAuth = Depends(get_jira_oauth),
) -> Dict[str, Any]:
    """Initiate Jira OAuth flow."""
    try:
        logger.info(
            "Initiating Jira OAuth flow with redirect_uri=%s state=%s",
            request.redirect_uri,
            request.state,
        )

        signed_state = (
            _sign_oauth_state(request.state)
            if getattr(request, "state", None)
            else None
        )
        auth_url = jira_oauth.get_authorization_url(
            redirect_uri=request.redirect_uri, state=signed_state
        )

        return {
            "status": "success",
            "authorization_url": auth_url,
            "message": "Jira OAuth flow initiated successfully",
            "debug_info": {
                "redirect_uri": request.redirect_uri,
                "state": request.state,
                "auth_url": auth_url,
            },
        }
    except Exception as exc:
        logger.exception("Jira OAuth initiation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate Jira OAuth flow: {str(exc)}",
        )


@router.get("/jira/callback")
async def jira_oauth_callback(
    request: Request,
    jira_oauth: JiraOAuth = Depends(get_jira_oauth),
) -> Dict[str, Any]:
    """Handle Jira OAuth callback."""
    try:
        # Extract OAuth params
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        # Verify signed state to get user_id
        user_id = _verify_oauth_state(state)
        if not user_id:
            raise HTTPException(
                status_code=400, detail="Invalid or missing OAuth state"
            )

        # If we have an authorization code, exchange it for tokens and save to database
        if code:
            try:
                # Create DB session and service
                from app.core.database import SessionLocal

                db = SessionLocal()
                integrations_service = IntegrationsService(db)

                from .integrations_schema import JiraSaveRequest
                from datetime import datetime

                # Prefer configured redirect URI so it exactly matches what's registered
                config = Config()
                redirect_uri = config("JIRA_REDIRECT_URI", default=None)

                # If not configured, build a fallback using the incoming request's scheme/host/port
                if not redirect_uri:
                    scheme = request.url.scheme or "http"
                    host = request.url.hostname or "localhost"
                    port = request.url.port

                    # Include port when it's non-standard for the scheme
                    if port and not (
                        (scheme == "http" and port == 80)
                        or (scheme == "https" and port == 443)
                    ):
                        host = f"{host}:{port}"

                    redirect_uri = (
                        f"{scheme}://{host}/api/v1/integrations/jira/callback"
                    )

                logger.info(
                    f"Using Jira redirect_uri for token exchange: {redirect_uri}"
                )

                save_request = JiraSaveRequest(
                    code=code,
                    redirect_uri=redirect_uri,
                    instance_name="Jira Integration",
                    user_id=user_id,
                    integration_type="jira",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )

                # Save the integration (this will exchange the code and persist tokens)
                save_result = await integrations_service.save_jira_integration(
                    save_request, user_id
                )

                db.close()

                # Redirect to frontend with success info
                config = Config()
                frontend_url = config("FRONTEND_URL", default="http://localhost:3000")

                # Ensure frontend_url has protocol
                if frontend_url and not frontend_url.startswith(
                    ("http://", "https://")
                ):
                    frontend_url = f"https://{frontend_url}"

                redirect_url = f"{frontend_url}/integrations/jira/redirect?success=true&integration_id={save_result.get('integration_id')}&user_name={save_result.get('user_name', '')}"

                return RedirectResponse(url=redirect_url)

            except Exception as e:
                logger.exception("Failed to save Jira integration")

                # Redirect to frontend with error
                config = Config()
                frontend_url = config("FRONTEND_URL", default="http://localhost:3000")

                if frontend_url and not frontend_url.startswith(
                    ("http://", "https://")
                ):
                    frontend_url = f"https://{frontend_url}"

                error_message = urllib.parse.quote(str(e), safe="")
                redirect_url = f"{frontend_url}/integrations/jira/redirect?error={error_message}&user_id={user_id}"

                return RedirectResponse(url=redirect_url)

        else:
            # No code provided — redirect to frontend with error
            config = Config()
            frontend_url = config("FRONTEND_URL", default="http://localhost:3000")

            if frontend_url and not frontend_url.startswith(("http://", "https://")):
                frontend_url = f"https://{frontend_url}"

            error_msg = "No authorization code received from Jira"
            if error:
                error_msg = f"OAuth error: {error}"
            elif error_description:
                error_msg = f"OAuth error: {error_description}"

            error_message = urllib.parse.quote(error_msg, safe="")
            redirect_url = f"{frontend_url}/integrations/jira/redirect?error={error_message}&user_id={user_id}"

            return RedirectResponse(url=redirect_url)

    except Exception as exc:
        logger.exception("Jira OAuth callback failed")
        raise HTTPException(
            status_code=400, detail=f"Jira OAuth callback failed: {str(exc)}"
        )


@router.get("/jira/status/{user_id}")
async def get_jira_status(
    user_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> JiraIntegrationStatus:
    """Get Jira integration status for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot access other users' integration status"
        )

    return await integrations_service.get_jira_integration_status(user_id)


@router.delete("/jira/revoke/{user_id}")
async def revoke_jira_access(
    user_id: str,
    jira_oauth: JiraOAuth = Depends(get_jira_oauth),
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Revoke Jira OAuth access for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot revoke other users' integrations"
        )

    tokens_removed = jira_oauth.revoke_access(user_id)
    deactivated = await integrations_service.deactivate_jira_integrations_for_user(
        user_id
    )

    if tokens_removed or deactivated:
        return {
            "status": "success",
            "message": "Jira access revoked successfully",
            "user_id": user_id,
            "deactivated_integrations": deactivated,
        }

    raise HTTPException(status_code=500, detail="Failed to revoke Jira access")


@router.get("/jira/{integration_id}/resources")
async def get_jira_resources(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """List Jira accessible resources for an integration."""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        result = await integrations_service.get_jira_accessible_resources(
            integration_id
        )
        return {
            "status": "success",
            "integration_id": integration_id,
            "resources": result.get("resources", []),
            "site_id": result.get("site_id"),
            "site_url": result.get("site_url"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching Jira accessible resources")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Jira accessible resources: {str(exc)}",
        )


@router.get("/jira/{integration_id}/projects")
async def get_jira_projects(
    integration_id: str,
    start_at: int = 0,
    max_results: int = 50,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Fetch Jira projects for an integration."""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        result = await integrations_service.get_jira_projects(
            integration_id, start_at=start_at, max_results=max_results
        )
        return {
            "status": "success",
            "integration_id": integration_id,
            "site_id": result.get("site_id"),
            "site_url": result.get("site_url"),
            "site_name": result.get("site_name"),
            "projects": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching Jira projects")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Jira projects: {str(exc)}",
        )


@router.get("/jira/{integration_id}/projects/{project_key}")
async def get_jira_project_details(
    integration_id: str,
    project_key: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Fetch details for a specific Jira project."""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        result = await integrations_service.get_jira_project_details(
            integration_id, project_key
        )
        return {
            "status": "success",
            "integration_id": integration_id,
            "project": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching Jira project details")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Jira project details: {str(exc)}",
        )


@router.post("/jira/save")
async def save_jira_integration(
    request: JiraSaveRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> JiraSaveResponse:
    """Save Jira integration after OAuth callback"""
    try:
        user_id = user["user_id"]
        result = await integrations_service.save_jira_integration(request, user_id)
        return JiraSaveResponse(success=True, data=result, error=None)
    except Exception as e:
        logger.exception("Error saving Jira integration")
        return JiraSaveResponse(
            success=False,
            data=None,
            error=f"Failed to save Jira integration: {str(e)}",
        )


@router.post("/confluence/initiate")
async def initiate_confluence_oauth(
    request: OAuthInitiateRequest,
    confluence_oauth: ConfluenceOAuth = Depends(get_confluence_oauth),
) -> Dict[str, Any]:
    """Initiate Confluence OAuth flow."""
    try:
        logger.info(
            "Initiating Confluence OAuth flow with redirect_uri=%s state=%s",
            request.redirect_uri,
            request.state,
        )
        signed_state = (
            _sign_oauth_state(request.state)
            if getattr(request, "state", None)
            else None
        )
        auth_url = confluence_oauth.get_authorization_url(
            redirect_uri=request.redirect_uri, state=signed_state
        )
        return {
            "status": "success",
            "authorization_url": auth_url,
            "message": "Confluence OAuth flow initiated successfully",
            "debug_info": {
                "redirect_uri": request.redirect_uri,
                "state": request.state,
                "auth_url": auth_url,
            },
        }
    except Exception as exc:
        logger.exception("Confluence OAuth initiation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate Confluence OAuth flow: {str(exc)}",
        )


@router.get("/confluence/callback")
async def confluence_oauth_callback(
    request: Request,
    confluence_oauth: ConfluenceOAuth = Depends(get_confluence_oauth),
) -> Dict[str, Any]:
    """Handle Confluence OAuth callback."""
    try:
        # Extract OAuth params
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        # Verify signed state to get user_id
        user_id = _verify_oauth_state(state)
        if not user_id:
            raise HTTPException(
                status_code=400, detail="Invalid or missing OAuth state"
            )

        # If we have an authorization code, exchange it for tokens and save to database
        if code:
            # Create DB session and service
            from app.core.database import SessionLocal

            db = SessionLocal()
            try:
                integrations_service = IntegrationsService(db)

                from .integrations_schema import ConfluenceSaveRequest
                from datetime import datetime

                # Prefer configured redirect URI so it exactly matches what's registered
                config = Config()
                redirect_uri = config("CONFLUENCE_REDIRECT_URI", default=None)

                # If not configured, build a fallback using the incoming request's scheme/host/port
                if not redirect_uri:
                    scheme = request.url.scheme
                    host = request.headers.get("host", "localhost")
                    redirect_uri = (
                        f"{scheme}://{host}/api/v1/integrations/confluence/callback"
                    )

                # Build the save request
                save_request = ConfluenceSaveRequest(
                    code=code,
                    redirect_uri=redirect_uri,
                    instance_name=f"Confluence-{datetime.now().strftime('%Y%m%d')}",
                    user_id=user_id,
                    integration_type="confluence",
                    timestamp=datetime.now().isoformat(),
                )

                # Save the integration
                result = await integrations_service.save_confluence_integration(
                    save_request, user_id
                )

                # Redirect to frontend with success
                frontend_url = config("FRONTEND_URL", default="http://localhost:3000")
                if frontend_url and not frontend_url.startswith(
                    ("http://", "https://")
                ):
                    frontend_url = f"https://{frontend_url}"

                redirect_url = f"{frontend_url}/integrations/confluence/redirect?success=true&user_id={user_id}&integration_id={result.get('integration_id', '')}"
                return RedirectResponse(url=redirect_url)

            except Exception as save_error:
                logger.exception("Error saving Confluence integration")
                # Redirect to frontend with error
                config = Config()
                frontend_url = config("FRONTEND_URL", default="http://localhost:3000")
                if frontend_url and not frontend_url.startswith(
                    ("http://", "https://")
                ):
                    frontend_url = f"https://{frontend_url}"

                error_message = urllib.parse.quote(str(save_error), safe="")
                redirect_url = f"{frontend_url}/integrations/confluence/redirect?error={error_message}&user_id={user_id}"
                return RedirectResponse(url=redirect_url)
            finally:
                # Always close the database session to prevent connection leaks
                db.close()
        else:
            # No code provided — redirect to frontend with error
            config = Config()
            frontend_url = config("FRONTEND_URL", default="http://localhost:3000")
            if frontend_url and not frontend_url.startswith(("http://", "https://")):
                frontend_url = f"https://{frontend_url}"

            error_msg = "No authorization code received from Confluence"
            if error:
                error_msg = f"OAuth error: {error}"
            elif error_description:
                error_msg = f"OAuth error: {error_description}"

            error_message = urllib.parse.quote(error_msg, safe="")
            redirect_url = f"{frontend_url}/integrations/confluence/redirect?error={error_message}&user_id={user_id}"
            return RedirectResponse(url=redirect_url)

    except Exception as exc:
        logger.exception("Confluence OAuth callback failed")
        raise HTTPException(
            status_code=400, detail=f"Confluence OAuth callback failed: {str(exc)}"
        )


@router.get("/confluence/status/{user_id}")
async def get_confluence_status(
    user_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> ConfluenceIntegrationStatus:
    """Get Confluence integration status for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot access other users' integration status"
        )
    return await integrations_service.get_confluence_integration_status(user_id)


@router.delete("/confluence/revoke/{user_id}")
async def revoke_confluence_access(
    user_id: str,
    confluence_oauth: ConfluenceOAuth = Depends(get_confluence_oauth),
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Revoke Confluence OAuth access for a user"""
    # Verify the authenticated user matches the requested user_id
    if user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot revoke other users' integrations"
        )

    tokens_removed = confluence_oauth.revoke_access(user_id)
    deactivated = (
        await integrations_service.deactivate_confluence_integrations_for_user(user_id)
    )

    if tokens_removed or deactivated:
        return {
            "status": "success",
            "message": "Confluence access revoked successfully",
            "user_id": user_id,
            "deactivated_integrations": deactivated,
        }
    raise HTTPException(status_code=500, detail="Failed to revoke Confluence access")


@router.get("/confluence/{integration_id}/resources")
async def get_confluence_resources(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Get accessible Confluence resources for an integration"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        result = await integrations_service.get_confluence_accessible_resources(
            integration_id
        )
        return {
            "status": "success",
            "resources": result.get("resources", []),
            "site_id": result.get("site_id"),
            "site_url": result.get("site_url"),
            "integration_id": integration_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error fetching Confluence resources for integration",
            integration_id=integration_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Confluence resources: {str(exc)}",
        )


@router.get("/confluence/{integration_id}/spaces")
async def get_confluence_spaces(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Get Confluence spaces for an integration"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        spaces = await integrations_service.get_confluence_spaces(integration_id)
        return {
            "status": "success",
            "spaces": spaces,
            "integration_id": integration_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            f"Error fetching Confluence spaces for integration {integration_id}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Confluence spaces: {str(exc)}",
        )


@router.post("/confluence/save")
async def save_confluence_integration(
    request: ConfluenceSaveRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> ConfluenceSaveResponse:
    """Save Confluence integration after OAuth callback"""
    try:
        # Use authenticated user id and pass to service
        user_id = user["user_id"]
        result = await integrations_service.save_confluence_integration(
            request, user_id
        )
        return ConfluenceSaveResponse(success=True, data=result, error=None)
    except Exception as e:
        logger.exception("Error saving Confluence integration")
        return ConfluenceSaveResponse(
            success=False,
            data=None,
            error=f"Failed to save Confluence integration: {str(e)}",
        )


@router.post("/sentry/webhook")
async def sentry_webhook(request: Request) -> Dict[str, Any]:
    """Handle Sentry webhook requests"""
    import json

    try:
        # Log the incoming webhook request details
        logger.info("Sentry webhook received")
        logger.info("Request details", method=request.method, url=str(request.url))
        sanitized_headers = sanitize_headers(dict(request.headers))
        logger.debug("Request headers", headers=sanitized_headers)

        # Get query parameters
        query_params = dict(request.query_params)
        params_summary = get_params_summary(query_params)
        logger.info(
            "Query parameters",
            params_keys=params_summary["keys"],
            params_count=params_summary["count"],
        )
        logger.debug("Query parameters", params_summary=params_summary)

        # Try to get request body if it exists
        webhook_data = {}
        try:
            body = await request.body()
            if body:
                body_text = body.decode("utf-8")
                body_summary = get_body_summary(body_text)
                logger.info("Request body received", body_length=body_summary["length"])
                logger.debug(
                    "Request body preview", body_preview=body_summary["preview"]
                )

                # Try to parse as JSON
                try:
                    webhook_data = json.loads(body_text)
                except json.JSONDecodeError:
                    logger.warning("Request body is not valid JSON")
                    webhook_data = {"raw_body": body_text}
            else:
                logger.info("Request body: (empty)")
        except Exception as e:
            logger.warning("Could not read request body", error=str(e))

        # Log form data if present
        try:
            form_data = await request.form()
            if form_data:
                form_dict = {k: str(v) for k, v in form_data.items()}
                form_summary = get_params_summary(form_dict)
                logger.info(
                    "Form data received",
                    form_fields=form_summary["keys"],
                    form_count=form_summary["count"],
                )
                logger.debug("Form data summary", form_summary=form_summary)
                webhook_data.update(form_dict)
        except Exception as e:
            logger.debug("No form data present", error=str(e))

        # Extract event type from headers or payload
        event_type = (
            dict(request.headers).get("X-Sentry-Event")
            or webhook_data.get("action")
            or "sentry.unknown"
        )

        # Get integration ID from query params or headers
        integration_id = query_params.get("integration_id") or dict(
            request.headers
        ).get("X-Integration-ID")

        if integration_id:
            # Initialize event bus and publish webhook event
            from app.modules.event_bus import CeleryEventBus
            from app.celery.celery_app import celery_app

            event_bus = CeleryEventBus(celery_app)

            try:
                event_id = await event_bus.publish_webhook_event(
                    integration_id=integration_id,
                    integration_type="sentry",
                    event_type=event_type,
                    payload=webhook_data,
                    headers=dict(request.headers),
                    source_ip=request.client.host if request.client else None,
                )

                logger.info(
                    f"Sentry webhook event {event_id} published for integration {integration_id}, "
                    f"type: {event_type}"
                )

                return {
                    "status": "success",
                    "message": "Sentry webhook logged and published to event bus",
                    "logged_at": time.time(),
                    "event_id": event_id,
                    "event_type": event_type,
                    "integration_id": integration_id,
                }
            except Exception as e:
                logger.exception("Failed to publish Sentry webhook to event bus")
                # Continue with normal response even if event bus fails
                return {
                    "status": "success",
                    "message": "Sentry webhook logged successfully (event bus failed)",
                    "logged_at": time.time(),
                    "event_bus_error": str(e),
                }
        else:
            logger.warning("No integration_id provided in Sentry webhook request")
            return {
                "status": "success",
                "message": "Sentry webhook logged successfully (no integration_id for event bus)",
                "logged_at": time.time(),
            }

    except Exception as e:
        logger.exception("Error processing Sentry webhook")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Sentry webhook: {str(e)}",
        )


@router.post("/linear/webhook")
async def linear_webhook(
    request: Request,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Handle Linear webhook requests"""
    import json

    try:
        # Log the incoming webhook request details
        logger.info("Linear webhook received")
        logger.info("Request details", method=request.method, url=str(request.url))
        sanitized_headers = sanitize_headers(dict(request.headers))
        logger.debug("Request headers", headers=sanitized_headers)

        # Get query parameters
        query_params = dict(request.query_params)
        params_summary = get_params_summary(query_params)
        logger.info(
            "Query parameters",
            params_keys=params_summary["keys"],
            params_count=params_summary["count"],
        )
        logger.debug("Query parameters", params_summary=params_summary)

        # Try to get request body if it exists
        webhook_data = {}
        try:
            body = await request.body()
            if body:
                body_text = body.decode("utf-8")
                body_summary = get_body_summary(body_text)
                logger.info("Request body received", body_length=body_summary["length"])
                logger.debug(
                    "Request body preview", body_preview=body_summary["preview"]
                )
                # Try to parse as JSON
                try:
                    webhook_data = json.loads(body_text)
                except json.JSONDecodeError:
                    webhook_data = {"raw_body": body_text}
            else:
                logger.info("Request body: (empty)")
        except Exception as e:
            logger.warning("Could not read request body", error=str(e))

        # Log form data if present
        try:
            form_data = await request.form()
            if form_data:
                form_dict = {k: str(v) for k, v in form_data.items()}
                form_summary = get_params_summary(form_dict)
                logger.info(
                    "Form data received",
                    form_fields=form_summary["keys"],
                    form_count=form_summary["count"],
                )
                logger.debug("Form data summary", form_summary=form_summary)
                webhook_data.update(form_dict)
        except Exception:
            pass

        # Use the service to log the webhook data
        result = await integrations_service.log_linear_webhook(webhook_data)

        # Extract event type from headers or payload
        event_type = (
            dict(request.headers).get("Linear-Event")
            or webhook_data.get("type", "").lower()
            or "linear.unknown"
        )

        # Get integration ID from Linear webhook payload
        # Linear webhooks include organizationId and webhookId in the payload
        organization_id = webhook_data.get("organizationId")
        webhook_id = webhook_data.get("webhookId")

        # Try to find integration by organization ID
        integration_id = None
        if organization_id:
            # Look up integration by org_id using unique_identifier
            integration = await integrations_service.get_linear_integration_by_org_id(
                organization_id
            )
            if integration:
                integration_id = integration.get("integration_id")

        # Fallback: try query params or headers (for manual testing)
        if not integration_id:
            integration_id = query_params.get("integration_id") or dict(
                request.headers
            ).get("X-Integration-ID")

        if integration_id:
            # Initialize event bus and publish webhook event
            from app.modules.event_bus import CeleryEventBus
            from app.celery.celery_app import celery_app

            event_bus = CeleryEventBus(celery_app)

            try:
                event_id = await event_bus.publish_webhook_event(
                    integration_id=integration_id,
                    integration_type="linear",
                    event_type=event_type,
                    payload=webhook_data,
                    headers=dict(request.headers),
                    source_ip=request.client.host if request.client else None,
                )

                return {
                    "status": "success",
                    "message": "Linear webhook logged and published to event bus",
                    "logged_at": time.time(),
                    "event_id": event_id,
                    "event_type": event_type,
                    "integration_id": integration_id,
                    "organization_id": organization_id,
                    "webhook_id": webhook_id,
                    "service_result": result,
                }
            except Exception as e:
                logger.exception("Event bus publishing failed")

                # Continue with normal response even if event bus fails
                return {
                    "status": "success",
                    "message": "Linear webhook logged successfully (event bus failed)",
                    "logged_at": time.time(),
                    "event_bus_error": str(e),
                    "organization_id": organization_id,
                    "webhook_id": webhook_id,
                    "service_result": result,
                }
        else:
            logger.warning(
                f"No Linear integration found for organization {organization_id}"
            )

            return {
                "status": "success",
                "message": "Linear webhook logged successfully (no matching integration found)",
                "logged_at": time.time(),
                "organization_id": organization_id,
                "webhook_id": webhook_id,
                "service_result": result,
            }

    except Exception as e:
        logger.exception("Error processing Linear webhook")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Linear webhook: {str(e)}",
        )


def verify_jira_webhook_jwt(
    authorization_header: Optional[str], client_secret: str
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Verify JWT token from Jira OAuth webhook"""
    if not authorization_header:
        logger.warning("No Authorization header in Jira webhook request")
        return False, None

    # Extract token from "Bearer <token>" format
    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.warning("Invalid Authorization header format: expected 'Bearer <token>'")
        return False, None

    token = parts[1]

    try:
        # Decode and verify JWT
        # Jira signs webhooks with HS256 (HMAC SHA-256) using client_secret
        decoded = jwt.decode(
            token,
            client_secret,
            algorithms=["HS256"],
            options={
                "verify_signature": True,
                "verify_exp": True,  # Verify expiration
                "verify_iat": True,  # Verify issued at
            },
        )

        return True, decoded

    except jwt.ExpiredSignatureError:
        logger.error("Jira webhook JWT has expired")
        return False, None
    except jwt.InvalidTokenError:
        logger.exception("Invalid Jira webhook JWT")
        return False, None
    except Exception:
        logger.exception("Error verifying Jira webhook JWT")
        return False, None


@router.post("/jira/webhook")
async def jira_webhook(
    request: Request,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Handle Jira webhook requests and publish to event bus"""
    import json

    try:
        # Log the incoming webhook request details
        logger.info("Jira webhook received")
        logger.info("Request details", method=request.method, url=str(request.url))
        sanitized_headers = sanitize_headers(dict(request.headers))
        logger.debug("Request headers", headers=sanitized_headers)

        # Get query parameters
        query_params = dict(request.query_params)
        params_summary = get_params_summary(query_params)
        logger.info(
            "Query parameters",
            params_keys=params_summary["keys"],
            params_count=params_summary["count"],
        )
        logger.debug("Query parameters", params_summary=params_summary)

        # Verify JWT authentication from Jira OAuth webhook
        config = Config(".env")
        jira_client_secret = config("JIRA_CLIENT_SECRET", default="")

        jwt_claims = None
        if jira_client_secret:
            auth_header = request.headers.get("Authorization")
            is_valid, jwt_claims = verify_jira_webhook_jwt(
                auth_header, jira_client_secret
            )
            if not is_valid:
                logger.warning(
                    "Jira webhook JWT verification failed - rejecting request"
                )
                raise HTTPException(
                    status_code=403, detail="Invalid or missing webhook authentication"
                )
        else:
            logger.warning(
                "JIRA_CLIENT_SECRET not configured - skipping JWT verification (INSECURE!)"
            )
            raise HTTPException(
                status_code=500,
                detail="Jira webhook cannot be processed: server misconfiguration",
            )

        # Try to read body
        webhook_data = {}
        try:
            body = await request.body()
            if body:
                body_text = body.decode("utf-8")
                body_summary = get_body_summary(body_text)
                logger.info("Request body received", body_length=body_summary["length"])
                logger.debug(
                    "Request body preview", body_preview=body_summary["preview"]
                )
                try:
                    webhook_data = json.loads(body_text)
                except json.JSONDecodeError:
                    webhook_data = {"raw_body": body_text}
            else:
                logger.info("Request body: (empty)")
        except Exception as e:
            logger.warning("Could not read request body", error=str(e))

        # Log form data if present
        try:
            form_data = await request.form()
            if form_data:
                form_dict = {k: str(v) for k, v in form_data.items()}
                form_summary = get_params_summary(form_dict)
                logger.info(
                    "Form data received",
                    form_fields=form_summary["keys"],
                    form_count=form_summary["count"],
                )
                logger.debug("Form data summary", form_summary=form_summary)
                webhook_data.update(form_dict)
        except Exception:
            pass

        # Use the service to log the webhook data
        result = await integrations_service.log_jira_webhook(webhook_data)

        # Determine event type
        event_type = (
            dict(request.headers).get("X-Event-Key")
            or webhook_data.get("webhookEvent")
            or "jira.unknown"
        )

        # Attempt to determine integration id from payload
        integration_id = None

        # OAuth webhooks include matchedWebhookIds - use this to look up the integration
        # IMPORTANT: webhook_id alone is not unique! Same webhook_id can exist across different Jira sites.
        # We need to match BOTH webhook_id AND site_id to find the correct integration.
        matched_webhook_ids = (
            webhook_data.get("matchedWebhookIds", [])
            if isinstance(webhook_data, dict)
            else []
        )

        # Extract site_id from multiple sources (in priority order)
        site_id = None
        if jwt_claims:
            # JWT claims (most reliable)
            context = jwt_claims.get("context", {})
            site_id = context.get("cloudId") if isinstance(context, dict) else None
            site_id = site_id or jwt_claims.get("aud")  # Fallback to audience

        if not site_id and isinstance(webhook_data, dict):
            # Webhook payload
            site_id = webhook_data.get("cloudId") or webhook_data.get("siteId")

        if not site_id:
            # HTTP headers
            site_id = dict(request.headers).get("X-Atlassian-Cloud-Id")

        # Try to find integration by BOTH webhook_id AND site_id (composite key)
        if matched_webhook_ids:
            try:
                from app.core.database import get_db
                from app.core.models import Integration

                db = next(get_db())
                try:
                    integrations = (
                        db.query(Integration)
                        .filter(
                            Integration.integration_type == "jira",
                            Integration.active == True,  # noqa: E712
                        )
                        .all()
                    )

                    for integration in integrations:
                        metadata = integration.integration_metadata or {}
                        webhooks = metadata.get("webhooks", [])

                        for webhook in webhooks:
                            webhook_id = webhook.get("id")
                            webhook_site_id = webhook.get("site_id")

                            # Match webhook_id from payload
                            if webhook_id in matched_webhook_ids:
                                # Verify site_id matches (composite key for uniqueness)
                                if site_id and webhook_site_id == site_id:
                                    integration_id = integration.integration_id
                                    logger.info(
                                        f"Found integration {integration_id} for webhook {webhook_id} + site {site_id}"
                                    )
                                    break
                                elif not site_id:
                                    # Fallback: match webhook_id only if site_id unavailable (less secure)
                                    integration_id = integration.integration_id
                                    logger.warning(
                                        f"Found integration {integration_id} for webhook {webhook_id} (no site_id verification)"
                                    )
                                    break

                        if integration_id:
                            break
                finally:
                    db.close()
            except Exception as e:
                logger.debug("Jira webhook ID lookup failed", error=str(e))

        # Fallback: Try site_id lookup via service (legacy/backup method)
        if not integration_id and site_id:
            try:
                integration = (
                    await integrations_service.check_existing_jira_integration(site_id)
                )
                if integration:
                    integration_id = integration.get("integration_id")
            except Exception as e:
                logger.debug("Jira webhook site lookup failed", error=str(e))

        # Fallback to query param or header
        query_params = dict(request.query_params)
        if not integration_id:
            integration_id = query_params.get("integration_id") or dict(
                request.headers
            ).get("X-Integration-ID")

        if integration_id:
            from app.modules.event_bus import CeleryEventBus
            from app.celery.celery_app import celery_app

            event_bus = CeleryEventBus(celery_app)

            try:
                event_id = await event_bus.publish_webhook_event(
                    integration_id=integration_id,
                    integration_type="jira",
                    event_type=event_type,
                    payload=webhook_data,
                    headers=dict(request.headers),
                    source_ip=request.client.host if request.client else None,
                )

                return {
                    "status": "success",
                    "message": "Jira webhook logged and published to event bus",
                    "logged_at": time.time(),
                    "event_id": event_id,
                    "event_type": event_type,
                    "integration_id": integration_id,
                    "service_result": result,
                }
            except Exception as e:
                logger.exception("Failed to publish Jira webhook to event bus")
                return {
                    "status": "success",
                    "message": "Jira webhook logged successfully (event bus failed)",
                    "logged_at": time.time(),
                    "event_bus_error": str(e),
                    "service_result": result,
                }

        logger.warning("No integration_id provided or found for Jira webhook request")
        return {
            "status": "success",
            "message": "Jira webhook logged successfully (no integration_id for event bus)",
            "logged_at": time.time(),
            "service_result": result,
        }

    except Exception as e:
        logger.exception("Error processing Jira webhook")
        raise HTTPException(
            status_code=500, detail=f"Failed to process Jira webhook: {str(e)}"
        )


@router.post("/sentry/save")
async def save_sentry_integration(
    request: SentrySaveRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> SentrySaveResponse:
    """Save Sentry integration after OAuth callback"""
    try:
        # Get the authenticated Potpie user's ID
        user_id = user["user_id"]
        result = await integrations_service.save_sentry_integration(request, user_id)
        return SentrySaveResponse(success=True, data=result, error=None)
    except Exception as e:
        logger.exception("Error saving Sentry integration")
        return SentrySaveResponse(
            success=False,
            data=None,
            error=f"Failed to save Sentry integration: {str(e)}",
        )


@router.post("/save", response_model=IntegrationSaveResponse)
async def save_integration(
    request: IntegrationSaveRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(get_api_key_user),
) -> IntegrationSaveResponse:
    """Save an integration with configurable and optional fields"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]
        result = await integrations_service.save_integration(request, user_id)
        return IntegrationSaveResponse(success=True, data=result, error=None)
    except Exception as e:
        logger.exception("Error saving integration")
        return IntegrationSaveResponse(
            success=False,
            data=None,
            error=f"Failed to save integration: {str(e)}",
        )


@router.get("/connected")
async def list_connected_integrations(
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """List all connected integrations that are saved for the authenticated user"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]

        # Get integrations only for this user
        user_integrations = await integrations_service.get_integrations_by_user(user_id)

        # Filter only active integrations
        connected_integrations = {
            integration_id: integration_data
            for integration_id, integration_data in user_integrations.items()
            if integration_data.get("active", False)
        }

        return {
            "status": "success",
            "count": len(connected_integrations),
            "connected_integrations": connected_integrations,
        }
    except Exception as e:
        logger.exception("Error listing connected integrations")
        raise HTTPException(
            status_code=500, detail=f"Failed to list connected integrations: {str(e)}"
        )


@router.get("/list")
async def list_integrations(
    integration_type: Optional[str] = None,
    org_slug: Optional[str] = None,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """List all integrations with optional filtering"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]

        # Get integrations for this user
        user_integrations = await integrations_service.get_integrations_by_user(user_id)

        # Apply additional filters if provided
        if integration_type:
            integrations = {
                integration_id: integration_data
                for integration_id, integration_data in user_integrations.items()
                if integration_data.get("integration_type") == integration_type
            }
        elif org_slug:
            integrations = {
                integration_id: integration_data
                for integration_id, integration_data in user_integrations.items()
                if integration_data.get("scope_data", {}).get("org_slug") == org_slug
            }
        else:
            integrations = user_integrations

        return {
            "status": "success",
            "count": len(integrations),
            "integrations": integrations,
        }
    except Exception as e:
        logger.exception("Error listing integrations")
        raise HTTPException(
            status_code=500, detail=f"Failed to list integrations: {str(e)}"
        )


@router.get("/{integration_id}")
async def get_integration(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Get a specific integration by ID (only if owned by the authenticated user)"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]

        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration:
            raise HTTPException(
                status_code=404, detail=f"Integration not found: {integration_id}"
            )

        # Verify ownership - user can only view their own integrations
        if integration.get("created_by") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this integration",
            )

        return {"status": "success", "integration": integration}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting integration")
        raise HTTPException(
            status_code=500, detail=f"Failed to get integration: {str(e)}"
        )


@router.delete("/{integration_id}")
async def delete_integration(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Delete an integration by ID (only if owned by the authenticated user)"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]

        # Validate integration_id format (basic UUID format check)
        if not integration_id or len(integration_id) < 10:
            raise HTTPException(status_code=400, detail="Invalid integration ID format")

        # Check if integration exists before attempting deletion
        existing_integration = await integrations_service.get_integration_by_id(
            integration_id
        )
        if not existing_integration:
            raise HTTPException(
                status_code=404, detail=f"Integration not found: {integration_id}"
            )

        # Verify ownership - user can only delete their own integrations
        if existing_integration.get("created_by") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to delete this integration",
            )

        # Log the deletion attempt
        logger.info(
            f"User {hash_user_id(user_id)} attempting to delete integration: {integration_id}"
        )

        success = await integrations_service.delete_integration(integration_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete integration: {integration_id}",
            )

        logger.info("Successfully deleted integration", integration_id=integration_id)
        return {
            "status": "success",
            "message": f"Integration deleted: {integration_id}",
            "deleted_integration": {
                "integration_id": integration_id,
                "name": existing_integration.get("name"),
                "type": existing_integration.get("integration_type"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting integration", integration_id=integration_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to delete integration: {str(e)}"
        )


@router.patch("/{integration_id}/status")
async def update_integration_status(
    integration_id: str,
    active: bool,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Update integration active status (only if owned by the authenticated user)"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]

        # Check if integration exists and verify ownership
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration:
            raise HTTPException(
                status_code=404, detail=f"Integration not found: {integration_id}"
            )

        # Verify ownership
        if integration.get("created_by") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to update this integration",
            )

        success = await integrations_service.update_integration_status(
            integration_id, active
        )
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Integration not found: {integration_id}"
            )

        return {
            "status": "success",
            "message": f"Integration status updated: {integration_id} -> active: {active}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating integration status")
        raise HTTPException(
            status_code=500, detail=f"Failed to update integration status: {str(e)}"
        )


# New endpoints using schema models
@router.post("/create", response_model=IntegrationResponse)
async def create_integration(
    request: IntegrationCreateRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> IntegrationResponse:
    """Create a new integration using schema models"""
    # Override created_by with authenticated user to prevent spoofing
    request.created_by = user["user_id"]
    return await integrations_service.create_integration(request)


@router.get("/schema/{integration_id}", response_model=IntegrationResponse)
async def get_integration_schema(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> IntegrationResponse:
    """Get integration by ID using schema models (only if owned by the authenticated user)"""
    # Get the authenticated user's ID
    user_id = user["user_id"]

    result = await integrations_service.get_integration_schema(integration_id)

    # Verify ownership if integration was found
    if result.success and result.data:
        if result.data.created_by != user_id:
            return IntegrationResponse(
                success=False,
                data=None,
                error="You don't have permission to view this integration",
            )

    return result


@router.put("/schema/{integration_id}", response_model=IntegrationResponse)
async def update_integration_schema(
    integration_id: str,
    request: IntegrationUpdateRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> IntegrationResponse:
    """Update integration using schema models - currently only allows name updates (only if owned by the authenticated user)"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]

        # Validate integration_id format
        if not integration_id or len(integration_id) < 10:
            return IntegrationResponse(
                success=False, data=None, error="Invalid integration ID format"
            )

        # Check ownership before updating
        existing = await integrations_service.get_integration_schema(integration_id)
        if existing.success and existing.data:
            if existing.data.created_by != user_id:
                return IntegrationResponse(
                    success=False,
                    data=None,
                    error="You don't have permission to update this integration",
                )

        # Log the update attempt
        logger.info(
            f"User {hash_user_id(user_id)} attempting to update integration name: {integration_id} to '{request.name}'"
        )

        result = await integrations_service.update_integration(integration_id, request)

        if result.success:
            logger.info(
                "Successfully updated integration name", integration_id=integration_id
            )
        else:
            logger.warning(
                "Failed to update integration name",
                integration_id=integration_id,
                error=result.error,
            )

        return result
    except Exception as e:
        logger.exception(
            "Error updating integration name", integration_id=integration_id
        )
        return IntegrationResponse(
            success=False, data=None, error=f"Failed to update integration: {str(e)}"
        )


@router.delete("/schema/{integration_id}", response_model=IntegrationResponse)
async def delete_integration_schema(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> IntegrationResponse:
    """Delete integration using schema models (only if owned by the authenticated user)"""
    try:
        # Get the authenticated user's ID
        user_id = user["user_id"]

        # Validate integration_id format (basic UUID format check)
        if not integration_id or len(integration_id) < 10:
            return IntegrationResponse(
                success=False, data=None, error="Invalid integration ID format"
            )

        # Check ownership before deleting
        existing = await integrations_service.get_integration_schema(integration_id)
        if existing.success and existing.data:
            if existing.data.created_by != user_id:
                return IntegrationResponse(
                    success=False,
                    data=None,
                    error="You don't have permission to delete this integration",
                )

        # Log the deletion attempt
        logger.info(
            f"User {hash_user_id(user_id)} attempting to delete integration (schema): {integration_id}"
        )

        result = await integrations_service.delete_integration_schema(integration_id)

        if result.success:
            logger.info(
                "Successfully deleted integration (schema)",
                integration_id=integration_id,
            )
        else:
            logger.warning(
                "Failed to delete integration (schema)",
                integration_id=integration_id,
                error=result.error,
            )

        return result
    except Exception as e:
        logger.exception(
            "Error deleting integration (schema)", integration_id=integration_id
        )
        return IntegrationResponse(
            success=False, data=None, error=f"Failed to delete integration: {str(e)}"
        )


@router.get("/schema/list", response_model=IntegrationListResponse)
async def list_integrations_schema(
    integration_type: Optional[IntegrationType] = None,
    status: Optional[IntegrationStatus] = None,
    active: Optional[bool] = None,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> IntegrationListResponse:
    """List integrations using schema models with filtering for the authenticated user"""
    # Get the authenticated user's ID
    user_id = user["user_id"]

    return await integrations_service.list_integrations_schema(
        integration_type=integration_type, status=status, active=active, user_id=user_id
    )


# Sentry API endpoints
@router.get("/sentry/{integration_id}/organizations")
async def get_sentry_organizations(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Get Sentry organizations for an integration"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        organizations = await integrations_service.get_sentry_organizations(
            integration_id
        )
        return {
            "status": "success",
            "organizations": organizations,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting Sentry organizations")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Sentry organizations: {str(e)}"
        )


@router.get("/sentry/{integration_id}/organizations/{org_slug}/projects")
async def get_sentry_projects(
    integration_id: str,
    org_slug: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Get Sentry projects for an organization"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        projects = await integrations_service.get_sentry_projects(
            integration_id, org_slug
        )
        return {
            "status": "success",
            "org_slug": org_slug,
            "projects": projects,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting Sentry projects")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Sentry projects: {str(e)}"
        )


@router.get(
    "/sentry/{integration_id}/organizations/{org_slug}/projects/{project_slug}/issues"
)
async def get_sentry_issues(
    integration_id: str,
    org_slug: str,
    project_slug: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Get Sentry issues for a project"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        issues = await integrations_service.get_sentry_issues(
            integration_id, org_slug, project_slug
        )
        return {
            "status": "success",
            "org_slug": org_slug,
            "project_slug": project_slug,
            "issues": issues,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting Sentry issues")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Sentry issues: {str(e)}"
        )


@router.post("/sentry/{integration_id}/api/{endpoint:path}")
async def make_sentry_api_call(
    integration_id: str,
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Make a custom API call to Sentry"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        result = await integrations_service.make_sentry_api_call(
            integration_id, f"/{endpoint}", method, data
        )
        return {
            "status": "success",
            "endpoint": endpoint,
            "method": method,
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error making Sentry API call")
        raise HTTPException(
            status_code=500, detail=f"Failed to make Sentry API call: {str(e)}"
        )


@router.post("/sentry/{integration_id}/refresh-token")
async def refresh_sentry_token(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Refresh expired Sentry access token"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        result = await integrations_service.refresh_sentry_token(integration_id)
        return {
            "status": "success",
            "message": "Token refreshed successfully",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error refreshing Sentry token")
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh Sentry token: {str(e)}"
        )


@router.get("/sentry/{integration_id}/token-status")
async def get_sentry_token_status(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
    user: dict = Depends(AuthService.check_auth),
) -> Dict[str, Any]:
    """Get the status of Sentry access token (valid/expired)"""
    try:
        # Verify the user owns this integration
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration or integration.get("created_by") != user["user_id"]:
            raise HTTPException(
                status_code=403, detail="Integration not found or access denied"
            )

        # Try to get a valid token (this will refresh if expired)
        access_token = await integrations_service.get_valid_sentry_token(integration_id)

        return {
            "status": "success",
            "token_status": "valid",
            "has_token": bool(access_token),
            "message": "Token is valid and ready for use",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error checking Sentry token status")
        return {
            "status": "error",
            "token_status": "invalid",
            "has_token": False,
            "error": str(e),
        }
