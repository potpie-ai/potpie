import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import HTTPException

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_schema import (
    LoginRequest,
    SSOLoginRequest,
    ConfirmLinkingRequest,
    UnlinkProviderRequest,
    SetPrimaryProviderRequest,
    UserAuthProvidersResponse,
    AuthProviderResponse,
    AccountResponse,
)
from app.modules.auth.auth_service import auth_handler
from app.modules.auth.unified_auth_service import UnifiedAuthService
from app.modules.users.user_schema import CreateUser
from app.modules.users.user_service import UserService
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.posthog_helper import PostHogClient

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", None)

auth_router = APIRouter()
load_dotenv(override=True)


async def send_slack_message(message: str):
    payload = {"text": message}
    if SLACK_WEBHOOK_URL:
        requests.post(SLACK_WEBHOOK_URL, json=payload)


class AuthAPI:
    @auth_router.post("/login")
    async def login(login_request: LoginRequest):
        email, password = login_request.email, login_request.password

        try:
            res = auth_handler.login(email=email, password=password)
            id_token = res.get("idToken")
            return JSONResponse(content={"token": id_token}, status_code=200)
        except ValueError:
            return JSONResponse(
                content={"error": "Invalid email or password"}, status_code=401
            )
        except HTTPException as he:
            return JSONResponse(
                content={"error": f"HTTP Error: {str(he)}"}, status_code=he.status_code
            )
        except Exception as e:
            return JSONResponse(content={"error": f"ERROR: {str(e)}"}, status_code=400)

    @auth_router.post("/signup")
    async def signup(request: Request, db: Session = Depends(get_db)):
        body = json.loads(await request.body())
        uid = body["uid"]
        oauth_token = body["accessToken"]
        user_service = UserService(db)
        user = user_service.get_user_by_uid(uid)
        if user:
            message, error = user_service.update_last_login(uid, oauth_token)
            if error:
                return Response(content=message, status_code=400)
            else:
                return Response(
                    content=json.dumps({"uid": uid, "exists": True}),
                    status_code=200,
                )
        else:
            first_login = datetime.utcnow()
            provider_info = body["providerData"][0]
            provider_info["access_token"] = oauth_token
            user = CreateUser(
                uid=uid,
                email=body["email"],
                display_name=body["displayName"],
                email_verified=body["emailVerified"],
                created_at=first_login,
                last_login_at=first_login,
                provider_info=provider_info,
                provider_username=body["providerUsername"],
            )
            uid, message, error = user_service.create_user(user)

            await send_slack_message(
                f"New signup: {body['email']} ({body['displayName']})"
            )

            PostHogClient().send_event(
                uid,
                "signup_event",
                {
                    "email": body["email"],
                    "display_name": body["displayName"],
                    "github_username": body["providerUsername"],
                },
            )
            if error:
                return Response(content=message, status_code=400)
            return Response(
                content=json.dumps({"uid": uid, "exists": False}),
                status_code=201,
            )
    
    # ===== Multi-Provider SSO Endpoints =====
    
    @auth_router.post("/sso/login")
    async def sso_login(
        request: Request,
        sso_request: SSOLoginRequest,
        db: Session = Depends(get_db),
    ):
        """
        SSO Login endpoint.
        
        Handles login via any SSO provider (Google, Azure, Okta, SAML).
        Returns one of three statuses:
        - 'success': User authenticated
        - 'needs_linking': User exists with different provider, needs confirmation
        - 'new_user': New user created
        """
        try:
            unified_auth = UnifiedAuthService(db)
            
            # Get request context
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            
            # Map SSO provider to our provider type
            provider_type = f"sso_{sso_request.sso_provider}"
            
            # TODO: Verify the ID token with the SSO provider
            # For now, we trust the frontend validation
            # In production, verify token server-side
            
            # Extract provider data from token (mock for now)
            provider_data = sso_request.provider_data or {}
            provider_uid = provider_data.get("sub") or provider_data.get("oid") or sso_request.email
            
            # Authenticate or create user
            user, response = unified_auth.authenticate_or_create(
                email=sso_request.email,
                provider_type=provider_type,
                provider_uid=provider_uid,
                provider_data=provider_data,
                display_name=provider_data.get("name") or sso_request.email.split("@")[0],
                email_verified=True,  # SSO providers always verify email
                ip_address=ip_address,
                user_agent=user_agent,
            )
            
            # Send Slack notification for new users
            if response.status == "new_user":
                await send_slack_message(
                    f"New SSO signup: {sso_request.email} via {sso_request.sso_provider}"
                )
                
                PostHogClient().send_event(
                    user.uid,
                    "sso_signup_event",
                    {
                        "email": sso_request.email,
                        "sso_provider": sso_request.sso_provider,
                    },
                )
            
            return JSONResponse(
                content=response.model_dump(),
                status_code=200 if response.status == "success" else 202,
            )
            
        except Exception as e:
            return JSONResponse(
                content={"error": f"SSO login failed: {str(e)}"},
                status_code=400,
            )
    
    @auth_router.post("/providers/confirm-linking")
    async def confirm_provider_linking(
        confirm_request: ConfirmLinkingRequest,
        db: Session = Depends(get_db),
    ):
        """
        Confirm linking a new provider to existing account.
        
        Called after user confirms they want to link the provider
        when 'needs_linking' status is returned from login.
        """
        try:
            unified_auth = UnifiedAuthService(db)
            
            new_provider = unified_auth.confirm_provider_link(
                confirm_request.linking_token
            )
            
            if not new_provider:
                return JSONResponse(
                    content={"error": "Invalid or expired linking token"},
                    status_code=400,
                )
            
            return JSONResponse(
                content={
                    "message": "Provider linked successfully",
                    "provider": AuthProviderResponse.from_orm(new_provider).model_dump(),
                },
                status_code=200,
            )
            
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to link provider: {str(e)}"},
                status_code=400,
            )
    
    @auth_router.delete("/providers/cancel-linking/{linking_token}")
    async def cancel_provider_linking(
        linking_token: str,
        db: Session = Depends(get_db),
    ):
        """Cancel a pending provider link"""
        try:
            unified_auth = UnifiedAuthService(db)
            success = unified_auth.cancel_pending_link(linking_token)
            
            if success:
                return JSONResponse(
                    content={"message": "Linking cancelled"},
                    status_code=200,
                )
            else:
                return JSONResponse(
                    content={"error": "Linking token not found"},
                    status_code=404,
                )
                
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to cancel linking: {str(e)}"},
                status_code=400,
            )
    
    @auth_router.get("/providers/me")
    async def get_my_providers(
        request: Request,
        db: Session = Depends(get_db),
    ):
        """
        Get all authentication providers for the current user.
        
        Requires authentication.
        """
        try:
            # Get user from auth token
            user_data = await auth_handler.check_auth(request, None)
            user_id = user_data.get("user_id")
            
            if not user_id:
                return JSONResponse(
                    content={"error": "Authentication required"},
                    status_code=401,
                )
            
            unified_auth = UnifiedAuthService(db)
            providers = unified_auth.get_user_providers(user_id)
            
            primary_provider = next(
                (p for p in providers if p.is_primary),
                None
            )
            
            response = UserAuthProvidersResponse(
                providers=[AuthProviderResponse.from_orm(p) for p in providers],
                primary_provider=AuthProviderResponse.from_orm(primary_provider) if primary_provider else None,
            )
            
            return JSONResponse(
                content=response.model_dump(),
                status_code=200,
            )
            
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to get providers: {str(e)}"},
                status_code=400,
            )
    
    @auth_router.post("/providers/set-primary")
    async def set_primary_provider(
        request: Request,
        primary_request: SetPrimaryProviderRequest,
        db: Session = Depends(get_db),
    ):
        """Set a provider as the primary login method"""
        try:
            # Get user from auth token
            user_data = await auth_handler.check_auth(request, None)
            user_id = user_data.get("user_id")
            
            if not user_id:
                return JSONResponse(
                    content={"error": "Authentication required"},
                    status_code=401,
                )
            
            unified_auth = UnifiedAuthService(db)
            success = unified_auth.set_primary_provider(
                user_id,
                primary_request.provider_type,
            )
            
            if success:
                return JSONResponse(
                    content={"message": "Primary provider updated"},
                    status_code=200,
                )
            else:
                return JSONResponse(
                    content={"error": "Provider not found"},
                    status_code=404,
                )
                
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to set primary provider: {str(e)}"},
                status_code=400,
            )
    
    @auth_router.delete("/providers/unlink")
    async def unlink_provider(
        request: Request,
        unlink_request: UnlinkProviderRequest,
        db: Session = Depends(get_db),
    ):
        """Unlink a provider from account"""
        try:
            # Get user from auth token
            user_data = await auth_handler.check_auth(request, None)
            user_id = user_data.get("user_id")
            
            if not user_id:
                return JSONResponse(
                    content={"error": "Authentication required"},
                    status_code=401,
                )
            
            unified_auth = UnifiedAuthService(db)
            
            try:
                success = unified_auth.unlink_provider(
                    user_id,
                    unlink_request.provider_type,
                )
                
                if success:
                    return JSONResponse(
                        content={"message": "Provider unlinked"},
                        status_code=200,
                    )
                else:
                    return JSONResponse(
                        content={"error": "Provider not found"},
                        status_code=404,
                    )
                    
            except ValueError as ve:
                # Cannot unlink last provider
                return JSONResponse(
                    content={"error": str(ve)},
                    status_code=400,
                )
                
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to unlink provider: {str(e)}"},
                status_code=400,
            )
    
    @auth_router.get("/account/me")
    async def get_my_account(
        request: Request,
        db: Session = Depends(get_db),
    ):
        """Get complete account information including all providers"""
        try:
            # Get user from auth token
            user_data = await auth_handler.check_auth(request, None)
            user_id = user_data.get("user_id")
            
            if not user_id:
                return JSONResponse(
                    content={"error": "Authentication required"},
                    status_code=401,
                )
            
            user_service = UserService(db)
            user = user_service.get_user_by_uid(user_id)
            
            if not user:
                return JSONResponse(
                    content={"error": "User not found"},
                    status_code=404,
                )
            
            unified_auth = UnifiedAuthService(db)
            providers = unified_auth.get_user_providers(user_id)
            
            primary_provider = next(
                (p.provider_type for p in providers if p.is_primary),
                None
            )
            
            response = AccountResponse(
                user_id=user.uid,
                email=user.email,
                display_name=user.display_name,
                organization=user.organization,
                organization_name=user.organization_name,
                email_verified=user.email_verified,
                created_at=user.created_at,
                providers=[AuthProviderResponse.from_orm(p) for p in providers],
                primary_provider=primary_provider,
            )
            
            return JSONResponse(
                content=response.model_dump(),
                status_code=200,
            )
            
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to get account: {str(e)}"},
                status_code=400,
            )
