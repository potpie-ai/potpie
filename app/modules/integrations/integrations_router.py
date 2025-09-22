from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.config import Config
from starlette.responses import RedirectResponse
from typing import Dict, Any, Optional, List
import logging
import time
import urllib.parse

from app.core.database import get_db
from app.api.router import get_api_key_user

from .sentry_oauth_v2 import SentryOAuthV2
from .linear_oauth import LinearOAuth
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
    IntegrationCreateRequest,
    IntegrationUpdateRequest,
    IntegrationResponse,
    IntegrationListResponse,
    IntegrationType,
    IntegrationStatus,
    IntegrationSaveRequest,
    IntegrationSaveResponse,
)

router = APIRouter(prefix="/integrations", tags=["integrations"])


def get_sentry_oauth() -> SentryOAuthV2:
    """Dependency to get Sentry OAuth integration instance"""
    config = Config()
    return SentryOAuthV2(config)


def get_linear_oauth() -> LinearOAuth:
    """Dependency to get Linear OAuth integration instance"""
    config = Config()
    return LinearOAuth(config)


def get_integrations_service(db: Session = Depends(get_db)) -> IntegrationsService:
    """Dependency to get integrations service instance with database session"""
    return IntegrationsService(db)


@router.post("/sentry/initiate")
async def initiate_sentry_oauth(
    request: OAuthInitiateRequest,
    sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth),
) -> Dict[str, Any]:
    """Initiate Sentry OAuth flow"""
    try:
        # Log the OAuth initiation details
        logging.info("=== OAuth Initiation Debug ===")
        logging.info(f"Redirect URI: {request.redirect_uri}")
        logging.info(f"State: {request.state}")

        # Generate authorization URL
        auth_url = sentry_oauth.get_authorization_url(
            redirect_uri=request.redirect_uri, state=request.state or None
        )

        logging.info(f"Generated authorization URL: {auth_url}")

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
        logging.error(f"OAuth initiation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate OAuth flow: {str(e)}"
        )


@router.get("/sentry/callback")
async def sentry_oauth_callback(
    request: Request,
    user_id: str,
    sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth),
) -> OAuthStatusResponse:
    """Handle Sentry OAuth callback"""
    try:
        result = sentry_oauth.handle_callback(request, user_id)
        return OAuthStatusResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {str(e)}")


@router.get("/sentry/status/{user_id}")
async def get_sentry_status(
    user_id: str, sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth)
) -> SentryIntegrationStatus:
    """Get Sentry integration status for a user"""
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
    user_id: str, sentry_oauth: SentryOAuthV2 = Depends(get_sentry_oauth)
) -> Dict[str, Any]:
    """Revoke Sentry OAuth access for a user"""
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
        logging.info("Sentry webhook redirect received")
        logging.info(f"Request method: {request.method}")
        logging.info(f"Request URL: {request.url}")
        logging.info(f"Request headers: {dict(request.headers)}")

        # Get query parameters
        query_params = dict(request.query_params)
        logging.info(f"Query parameters: {query_params}")

        # Try to get request body if it exists
        try:
            body = await request.body()
            if body:
                logging.info(f"Request body: {body.decode('utf-8')}")
            else:
                logging.info("Request body: (empty)")
        except Exception as e:
            logging.warning(f"Could not read request body: {str(e)}")

        # Log form data if present
        try:
            form_data = await request.form()
            if form_data:
                logging.info(f"Form data: {dict(form_data)}")
        except Exception as e:
            logging.info(f"No form data present: {str(e)}")

        return {
            "status": "success",
            "message": "Sentry webhook redirect logged successfully",
            "logged_at": time.time(),
        }

    except Exception as e:
        logging.error(f"Error processing Sentry webhook redirect: {str(e)}")
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

        # Get scope parameter if provided (default to read)
        scope = request.query_params.get("scope", "read")

        # Generate authorization URL
        auth_url = linear_oauth.get_authorization_url(
            redirect_uri=redirect_uri, state=state, scope=scope
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
    user_id: Optional[str] = None,
    linear_oauth: LinearOAuth = Depends(get_linear_oauth),
):
    """Handle Linear OAuth callback"""
    try:
        # Extract OAuth parameters
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        # If no user_id provided, use a default or extract from state
        if not user_id:
            # Try to get user_id from state parameter if it contains user info
            if state and state != "SECURE_RANDOM":
                # If state contains user info, extract it
                user_id = state
            else:
                # Use a default user_id for now
                user_id = "default_user"
                logging.warning("No user_id provided in callback, using default_user")

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
                    save_request
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
                logging.error(f"Failed to save Linear integration: {str(e)}")

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
        logging.error(f"Linear OAuth callback failed: {str(e)}")

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
    user_id: str, linear_oauth: LinearOAuth = Depends(get_linear_oauth)
) -> LinearIntegrationStatus:
    """Get Linear integration status for a user"""
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
    user_id: str, linear_oauth: LinearOAuth = Depends(get_linear_oauth)
) -> Dict[str, Any]:
    """Revoke Linear OAuth access for a user"""
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
) -> LinearSaveResponse:
    """Save Linear integration after OAuth callback"""
    try:
        result = await integrations_service.save_linear_integration(request)
        return LinearSaveResponse(success=True, data=result, error=None)
    except Exception as e:
        logging.error(f"Error saving Linear integration: {str(e)}")
        return LinearSaveResponse(
            success=False,
            data=None,
            error=f"Failed to save Linear integration: {str(e)}",
        )


@router.post("/sentry/webhook")
async def sentry_webhook(request: Request) -> Dict[str, Any]:
    """Handle Sentry webhook requests"""
    import json

    try:
        # Log the incoming webhook request details
        logging.info("Sentry webhook received")
        logging.info(f"Request method: {request.method}")
        logging.info(f"Request URL: {request.url}")
        logging.info(f"Request headers: {dict(request.headers)}")

        # Get query parameters
        query_params = dict(request.query_params)
        logging.info(f"Query parameters: {query_params}")

        # Try to get request body if it exists
        webhook_data = {}
        try:
            body = await request.body()
            if body:
                body_text = body.decode("utf-8")
                logging.info(f"Request body: {body_text}")

                # Try to parse as JSON
                try:
                    webhook_data = json.loads(body_text)
                except json.JSONDecodeError:
                    logging.warning("Request body is not valid JSON")
                    webhook_data = {"raw_body": body_text}
            else:
                logging.info("Request body: (empty)")
        except Exception as e:
            logging.warning(f"Could not read request body: {str(e)}")

        # Log form data if present
        try:
            form_data = await request.form()
            if form_data:
                form_dict = {k: str(v) for k, v in form_data.items()}
                logging.info(f"Form data: {form_dict}")
                webhook_data.update(form_dict)
        except Exception as e:
            logging.info(f"No form data present: {str(e)}")

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

                logging.info(
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
                logging.error(
                    f"Failed to publish Sentry webhook to event bus: {str(e)}"
                )
                # Continue with normal response even if event bus fails
                return {
                    "status": "success",
                    "message": "Sentry webhook logged successfully (event bus failed)",
                    "logged_at": time.time(),
                    "event_bus_error": str(e),
                }
        else:
            logging.warning("No integration_id provided in Sentry webhook request")
            return {
                "status": "success",
                "message": "Sentry webhook logged successfully (no integration_id for event bus)",
                "logged_at": time.time(),
            }

    except Exception as e:
        logging.error(f"Error processing Sentry webhook: {str(e)}")
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
        # Get query parameters
        query_params = dict(request.query_params)

        # Try to get request body if it exists
        webhook_data = {}
        try:
            body = await request.body()
            if body:
                body_text = body.decode("utf-8")
                # Try to parse as JSON
                try:
                    webhook_data = json.loads(body_text)
                except json.JSONDecodeError:
                    webhook_data = {"raw_body": body_text}
        except Exception as e:
            logging.warning(f"Could not read request body: {str(e)}")

        # Log form data if present
        try:
            form_data = await request.form()
            if form_data:
                form_dict = {k: str(v) for k, v in form_data.items()}
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
                logging.error(f"Event bus publishing failed: {str(e)}")

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
            logging.warning(
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
        logging.error(f"Error processing Linear webhook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Linear webhook: {str(e)}",
        )


@router.post("/sentry/save")
async def save_sentry_integration(
    request: SentrySaveRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> SentrySaveResponse:
    """Save Sentry integration after OAuth callback"""
    try:
        result = await integrations_service.save_sentry_integration(request)
        return SentrySaveResponse(success=True, data=result, error=None)
    except Exception as e:
        logging.error(f"Error saving Sentry integration: {str(e)}")
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
        logging.error(f"Error saving integration: {str(e)}")
        return IntegrationSaveResponse(
            success=False,
            data=None,
            error=f"Failed to save integration: {str(e)}",
        )


@router.get("/connected")
async def list_connected_integrations(
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """List all connected integrations that are saved"""
    try:
        all_integrations = await integrations_service.get_all_integrations()

        # Filter only active integrations
        connected_integrations = {
            integration_id: integration_data
            for integration_id, integration_data in all_integrations.items()
            if integration_data.get("active", False)
        }

        return {
            "status": "success",
            "count": len(connected_integrations),
            "connected_integrations": connected_integrations,
        }
    except Exception as e:
        logging.error(f"Error listing connected integrations: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list connected integrations: {str(e)}"
        )


@router.get("/list")
async def list_integrations(
    integration_type: Optional[str] = None,
    org_slug: Optional[str] = None,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """List all integrations with optional filtering"""
    try:
        if integration_type:
            integrations = await integrations_service.get_integrations_by_type(
                integration_type
            )
        elif org_slug:
            integrations = await integrations_service.get_integrations_by_org_slug(
                org_slug
            )
        else:
            integrations = await integrations_service.get_all_integrations()

        return {
            "status": "success",
            "count": len(integrations),
            "integrations": integrations,
        }
    except Exception as e:
        logging.error(f"Error listing integrations: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list integrations: {str(e)}"
        )


@router.get("/{integration_id}")
async def get_integration(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Get a specific integration by ID"""
    try:
        integration = await integrations_service.get_integration_by_id(integration_id)
        if not integration:
            raise HTTPException(
                status_code=404, detail=f"Integration not found: {integration_id}"
            )

        return {"status": "success", "integration": integration}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting integration: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get integration: {str(e)}"
        )


@router.delete("/{integration_id}")
async def delete_integration(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Delete an integration by ID"""
    try:
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

        # Log the deletion attempt
        logging.info(f"Attempting to delete integration: {integration_id}")

        success = await integrations_service.delete_integration(integration_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete integration: {integration_id}",
            )

        logging.info(f"Successfully deleted integration: {integration_id}")
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
        logging.error(f"Error deleting integration {integration_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete integration: {str(e)}"
        )


@router.patch("/{integration_id}/status")
async def update_integration_status(
    integration_id: str,
    active: bool,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Update integration active status"""
    try:
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
        logging.error(f"Error updating integration status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update integration status: {str(e)}"
        )


# New endpoints using schema models
@router.post("/create", response_model=IntegrationResponse)
async def create_integration(
    request: IntegrationCreateRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> IntegrationResponse:
    """Create a new integration using schema models"""
    return await integrations_service.create_integration(request)


@router.get("/schema/{integration_id}", response_model=IntegrationResponse)
async def get_integration_schema(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> IntegrationResponse:
    """Get integration by ID using schema models"""
    return await integrations_service.get_integration_schema(integration_id)


@router.put("/schema/{integration_id}", response_model=IntegrationResponse)
async def update_integration_schema(
    integration_id: str,
    request: IntegrationUpdateRequest,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> IntegrationResponse:
    """Update integration using schema models - currently only allows name updates"""
    try:
        # Validate integration_id format
        if not integration_id or len(integration_id) < 10:
            return IntegrationResponse(
                success=False, data=None, error="Invalid integration ID format"
            )

        # Log the update attempt
        logging.info(
            f"Attempting to update integration name: {integration_id} to '{request.name}'"
        )

        result = await integrations_service.update_integration(integration_id, request)

        if result.success:
            logging.info(f"Successfully updated integration name: {integration_id}")
        else:
            logging.warning(
                f"Failed to update integration name: {integration_id} - {result.error}"
            )

        return result
    except Exception as e:
        logging.error(f"Error updating integration name {integration_id}: {str(e)}")
        return IntegrationResponse(
            success=False, data=None, error=f"Failed to update integration: {str(e)}"
        )


@router.delete("/schema/{integration_id}", response_model=IntegrationResponse)
async def delete_integration_schema(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> IntegrationResponse:
    """Delete integration using schema models"""
    try:
        # Validate integration_id format (basic UUID format check)
        if not integration_id or len(integration_id) < 10:
            return IntegrationResponse(
                success=False, data=None, error="Invalid integration ID format"
            )

        # Log the deletion attempt
        logging.info(f"Attempting to delete integration (schema): {integration_id}")

        result = await integrations_service.delete_integration_schema(integration_id)

        if result.success:
            logging.info(f"Successfully deleted integration (schema): {integration_id}")
        else:
            logging.warning(
                f"Failed to delete integration (schema): {integration_id} - {result.error}"
            )

        return result
    except Exception as e:
        logging.error(f"Error deleting integration (schema) {integration_id}: {str(e)}")
        return IntegrationResponse(
            success=False, data=None, error=f"Failed to delete integration: {str(e)}"
        )


@router.delete("/bulk")
async def delete_integrations_bulk(
    integration_ids: List[str],
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Delete multiple integrations by their IDs"""
    try:
        if not integration_ids:
            raise HTTPException(status_code=400, detail="No integration IDs provided")

        if len(integration_ids) > 100:  # Prevent bulk deletion of too many items
            raise HTTPException(
                status_code=400,
                detail="Cannot delete more than 100 integrations at once",
            )

        # Validate all integration IDs
        for integration_id in integration_ids:
            if not integration_id or len(integration_id) < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid integration ID format: {integration_id}",
                )

        logging.info(f"Attempting bulk deletion of {len(integration_ids)} integrations")

        results = {
            "successful_deletions": [],
            "failed_deletions": [],
            "total_requested": len(integration_ids),
            "total_successful": 0,
            "total_failed": 0,
        }

        # Process each integration deletion
        for integration_id in integration_ids:
            try:
                success = await integrations_service.delete_integration(integration_id)
                if success:
                    results["successful_deletions"].append(integration_id)
                    results["total_successful"] += 1
                else:
                    results["failed_deletions"].append(
                        {
                            "integration_id": integration_id,
                            "error": "Integration not found or deletion failed",
                        }
                    )
                    results["total_failed"] += 1
            except Exception as e:
                results["failed_deletions"].append(
                    {"integration_id": integration_id, "error": str(e)}
                )
                results["total_failed"] += 1

        logging.info(
            f"Bulk deletion completed: {results['total_successful']} successful, {results['total_failed']} failed"
        )

        return {
            "status": "completed",
            "message": f"Bulk deletion completed: {results['total_successful']} successful, {results['total_failed']} failed",
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in bulk deletion: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform bulk deletion: {str(e)}"
        )


@router.get("/schema/list", response_model=IntegrationListResponse)
async def list_integrations_schema(
    integration_type: Optional[IntegrationType] = None,
    status: Optional[IntegrationStatus] = None,
    active: Optional[bool] = None,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> IntegrationListResponse:
    """List integrations using schema models with filtering"""
    return await integrations_service.list_integrations_schema(
        integration_type=integration_type, status=status, active=active
    )


# Sentry API endpoints
@router.get("/sentry/{integration_id}/organizations")
async def get_sentry_organizations(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Get Sentry organizations for an integration"""
    try:
        organizations = await integrations_service.get_sentry_organizations(
            integration_id
        )
        return {
            "status": "success",
            "organizations": organizations,
        }
    except Exception as e:
        logging.error(f"Error getting Sentry organizations: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Sentry organizations: {str(e)}"
        )


@router.get("/sentry/{integration_id}/organizations/{org_slug}/projects")
async def get_sentry_projects(
    integration_id: str,
    org_slug: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Get Sentry projects for an organization"""
    try:
        projects = await integrations_service.get_sentry_projects(
            integration_id, org_slug
        )
        return {
            "status": "success",
            "org_slug": org_slug,
            "projects": projects,
        }
    except Exception as e:
        logging.error(f"Error getting Sentry projects: {str(e)}")
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
) -> Dict[str, Any]:
    """Get Sentry issues for a project"""
    try:
        issues = await integrations_service.get_sentry_issues(
            integration_id, org_slug, project_slug
        )
        return {
            "status": "success",
            "org_slug": org_slug,
            "project_slug": project_slug,
            "issues": issues,
        }
    except Exception as e:
        logging.error(f"Error getting Sentry issues: {str(e)}")
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
) -> Dict[str, Any]:
    """Make a custom API call to Sentry"""
    try:
        result = await integrations_service.make_sentry_api_call(
            integration_id, f"/{endpoint}", method, data
        )
        return {
            "status": "success",
            "endpoint": endpoint,
            "method": method,
            "data": result,
        }
    except Exception as e:
        logging.error(f"Error making Sentry API call: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to make Sentry API call: {str(e)}"
        )


@router.post("/sentry/{integration_id}/refresh-token")
async def refresh_sentry_token(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Refresh expired Sentry access token"""
    try:
        result = await integrations_service.refresh_sentry_token(integration_id)
        return {
            "status": "success",
            "message": "Token refreshed successfully",
            "data": result,
        }
    except Exception as e:
        logging.error(f"Error refreshing Sentry token: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh Sentry token: {str(e)}"
        )


@router.get("/sentry/{integration_id}/token-status")
async def get_sentry_token_status(
    integration_id: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Get the status of Sentry access token (valid/expired)"""
    try:
        # Try to get a valid token (this will refresh if expired)
        access_token = await integrations_service.get_valid_sentry_token(integration_id)

        return {
            "status": "success",
            "token_status": "valid",
            "has_token": bool(access_token),
            "message": "Token is valid and ready for use",
        }
    except Exception as e:
        logging.error(f"Error checking Sentry token status: {str(e)}")
        return {
            "status": "error",
            "token_status": "invalid",
            "has_token": False,
            "error": str(e),
        }


@router.get("/debug/oauth-config")
async def debug_oauth_config(
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Debug endpoint to check OAuth configuration"""
    try:
        validation_result = await integrations_service.validate_oauth_configuration()

        # Additional debugging information
        from starlette.config import Config

        config = Config()

        client_id = config("SENTRY_CLIENT_ID", default="")
        client_secret = config("SENTRY_CLIENT_SECRET", default="")
        redirect_uri = config("SENTRY_REDIRECT_URI", default="")

        debug_info = {
            "status": "success",
            "validation": validation_result,
            "raw_config": {
                "client_id": {
                    "value": client_id,
                    "length": len(client_id),
                    "preview": (
                        client_id[:8] + "..." if len(client_id) > 8 else client_id
                    ),
                    "is_hex": (
                        all(c in "0123456789abcdef" for c in client_id.lower())
                        if client_id
                        else False
                    ),
                },
                "client_secret": {
                    "value": client_secret,
                    "length": len(client_secret),
                    "preview": (
                        client_secret[:8] + "..."
                        if len(client_secret) > 8
                        else client_secret
                    ),
                    "is_hex": (
                        all(c in "0123456789abcdef" for c in client_secret.lower())
                        if client_secret
                        else False
                    ),
                },
                "redirect_uri": {
                    "value": redirect_uri,
                    "length": len(redirect_uri),
                    "is_url": (
                        redirect_uri.startswith(("http://", "https://"))
                        if redirect_uri
                        else False
                    ),
                },
            },
            "environment_check": {
                "python_version": __import__("sys").version,
                "httpx_available": True,
                "starlette_config_available": True,
            },
        }

        return debug_info
    except Exception as e:
        logging.error(f"Error checking OAuth config: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
        }


@router.post("/debug/test-token-exchange")
async def debug_test_token_exchange(
    code: str,
    redirect_uri: str,
    integrations_service: IntegrationsService = Depends(get_integrations_service),
) -> Dict[str, Any]:
    """Debug endpoint to test OAuth token exchange with detailed logging"""
    try:
        logging.info("=== DEBUG TOKEN EXCHANGE TEST ===")
        logging.info(f"Test code: {code}")
        logging.info(f"Test redirect URI: {redirect_uri}")

        # Call the token exchange method directly
        result = await integrations_service._exchange_code_for_tokens(
            code, redirect_uri
        )

        return {
            "status": "success",
            "message": "Token exchange test completed",
            "result": result,
        }
    except Exception as e:
        logging.error(f"Debug token exchange failed: {str(e)}")
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


@router.get("/debug/sentry-app-info")
async def debug_sentry_app_info() -> Dict[str, Any]:
    """Debug endpoint to provide Sentry app configuration guidance"""
    return {
        "status": "success",
        "sentry_app_setup": {
            "oauth_scopes": ["project:read", "org:read", "event:read", "issue:read"],
            "redirect_uri_requirements": {
                "current_configured": "http://localhost:3000/integrations/sentry/redirect",
                "must_match_exactly": True,
                "common_issues": [
                    "Trailing slash differences",
                    "Protocol mismatch (http vs https)",
                    "Port number differences",
                    "Case sensitivity",
                ],
            },
            "oauth_flow_steps": [
                "1. User clicks 'Connect to Sentry'",
                "2. Redirected to Sentry OAuth with redirect_uri",
                "3. User authorizes the app",
                "4. Sentry redirects back with authorization code",
                "5. Backend exchanges code for tokens using same redirect_uri",
            ],
            "troubleshooting": {
                "invalid_grant_causes": [
                    "Authorization code expired (10 minutes)",
                    "Code already used",
                    "Redirect URI mismatch",
                    "Wrong client credentials",
                ],
                "verification_steps": [
                    "Check Sentry app settings for exact redirect URI",
                    "Verify client ID and secret match",
                    "Ensure OAuth scopes are configured",
                    "Test with fresh authorization code",
                ],
            },
        },
    }
