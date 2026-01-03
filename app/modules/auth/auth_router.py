import json
import logging
import os

import requests
from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import HTTPException

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

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
from app.modules.auth.auth_service import auth_handler, AuthService
from app.modules.auth.auth_provider_model import UserAuthProvider
from app.modules.auth.auth_schema import AuthProviderCreate
from app.modules.auth.unified_auth_service import (
    UnifiedAuthService,
    PROVIDER_TYPE_FIREBASE_GITHUB,
    PROVIDER_TYPE_FIREBASE_EMAIL,
)
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
        """
        SIMPLIFIED Signup/Login endpoint.

        Two main flows:
        1. GitHub LINKING (linkToUserId provided):
           - Find existing user by linkToUserId (SSO UID)
           - Add GitHub provider with githubFirebaseUid as provider_uid

        2. Normal GitHub sign-in (no linkToUserId):
           - Check if GitHub UID already linked to a user
           - If yes: Return that user
           - If no: Create new user with GitHub UID
        """
        body = json.loads(await request.body())

        # Extract fields
        uid = body.get("uid")
        email = body.get("email")
        display_name = body.get("displayName") or body.get("display_name")
        email_verified = body.get("emailVerified") or body.get("email_verified", False)
        link_to_user_id = body.get("linkToUserId")  # SSO user UID to link GitHub to
        github_firebase_uid = body.get("githubFirebaseUid")  # GitHub's Firebase UID
        oauth_token = body.get("accessToken") or body.get("access_token")
        provider_username = body.get("providerUsername")

        # Extract provider info
        provider_info = {}
        if "providerData" in body:
            if isinstance(body["providerData"], list) and len(body["providerData"]) > 0:
                provider_info = (
                    body["providerData"][0].copy()
                    if isinstance(body["providerData"][0], dict)
                    else {}
                )
            elif isinstance(body["providerData"], dict):
                provider_info = body["providerData"].copy()

        # Add GitHub username to provider_info
        if provider_username:
            provider_info["username"] = provider_username

        logger.info(
            f"Signup: uid={uid}, email={email}, linkToUserId={link_to_user_id}, "
            f"githubFirebaseUid={github_firebase_uid}, hasToken={bool(oauth_token)}"
        )

        # Validate
        if not uid:
            return Response(
                content=json.dumps({"error": "Missing uid"}), status_code=400
            )
        if not email:
            return Response(
                content=json.dumps({"error": "Missing email"}), status_code=400
            )

        user_service = UserService(db)
        unified_auth = UnifiedAuthService(db)

        # Determine if this is GitHub flow
        is_github_flow = bool(oauth_token and provider_username)
        provider_type = (
            PROVIDER_TYPE_FIREBASE_GITHUB
            if is_github_flow
            else PROVIDER_TYPE_FIREBASE_EMAIL
        )

        # For GitHub: provider_uid is the GitHub Firebase UID
        # This is what we store in user_auth_providers.provider_uid
        provider_uid = github_firebase_uid or uid

        # ============================================================
        # FLOW 1: GITHUB LINKING (linkToUserId provided)
        # User has SSO account, wants to link GitHub
        # ============================================================
        if link_to_user_id and is_github_flow:
            logger.info(f"GitHub linking: Linking GitHub to SSO user {link_to_user_id}")

            # Find the SSO user
            db.expire_all()  # Ensure fresh data
            user = user_service.get_user_by_uid(link_to_user_id)

            if not user:
                logger.error(f"SSO user {link_to_user_id} not found in database!")
                return Response(
                    content=json.dumps(
                        {"error": "User not found. Please sign in again."}
                    ),
                    status_code=404,
                )

            logger.info(f"Found SSO user: uid={user.uid}, email={user.email}")

            # Check if GitHub already linked
            existing_github = (
                db.query(UserAuthProvider)
                .filter(
                    UserAuthProvider.user_id == user.uid,
                    UserAuthProvider.provider_type == PROVIDER_TYPE_FIREBASE_GITHUB,
                )
                .first()
            )

            if existing_github:
                logger.info(f"GitHub already linked to user {user.uid}")
                return Response(
                    content=json.dumps(
                        {
                            "uid": user.uid,
                            "exists": True,
                            "needs_github_linking": False,
                        }
                    ),
                    status_code=200,
                )

            # Check if this GitHub account is already linked to a different user
            existing_provider_with_uid = (
                db.query(UserAuthProvider)
                .filter(
                    UserAuthProvider.provider_type == PROVIDER_TYPE_FIREBASE_GITHUB,
                    UserAuthProvider.provider_uid == provider_uid,
                )
                .first()
            )

            if (
                existing_provider_with_uid
                and existing_provider_with_uid.user_id != user.uid
            ):
                error_message = (
                    "GitHub account is already linked to another account. "
                    "Please use a different GitHub account or contact support if you believe this is an error."
                )
                logger.warning(
                    f"GitHub account {provider_uid} is already linked to user {existing_provider_with_uid.user_id}, "
                    f"cannot link to user {user.uid}"
                )
                return Response(
                    content=json.dumps(
                        {
                            "error": error_message,
                            "details": f"GitHub account {provider_uid} is already linked to user {existing_provider_with_uid.user_id}, cannot link to user {user.uid}",
                        }
                    ),
                    status_code=409,  # Conflict
                )

            # Link GitHub provider
            logger.info(
                f"Linking GitHub (provider_uid={provider_uid}) to user {user.uid}"
            )
            provider_create = AuthProviderCreate(
                provider_type=provider_type,  # Use the variable instead of hardcoding
                provider_uid=provider_uid,  # GitHub Firebase UID
                provider_data=provider_info,
                access_token=oauth_token,
                is_primary=False,  # SSO is primary
            )

            try:
                unified_auth.add_provider(
                    user_id=user.uid, provider_create=provider_create
                )
                db.commit()
            except IntegrityError as e:
                db.rollback()
                # Check if it's a unique constraint violation for provider_uid
                error_str = str(e).lower()
                if "unique_provider_uid" in error_str or "uniqueviolation" in error_str:
                    error_message = (
                        "GitHub account is already linked to another account. "
                        "Please use a different GitHub account or contact support if you believe this is an error."
                    )
                    logger.error(
                        f"GitHub account {provider_uid} is already linked to another user: {e}"
                    )
                    return Response(
                        content=json.dumps(
                            {
                                "error": error_message,
                                "details": f"GitHub account {provider_uid} is already linked to another user. Database constraint violation: {str(e)}",
                            }
                        ),
                        status_code=409,  # Conflict
                    )
                # Re-raise other IntegrityErrors (e.g., unique_user_provider)
                raise
            except Exception as e:
                db.rollback()
                logger.error(
                    f"Unexpected error linking GitHub provider: {e}", exc_info=True
                )
                raise

            logger.info(f"Successfully linked GitHub to user {user.uid}")
            return Response(
                content=json.dumps(
                    {
                        "uid": user.uid,
                        "exists": True,
                        "needs_github_linking": False,
                    }
                ),
                status_code=200,
            )

        # ============================================================
        # FLOW 2: GITHUB SIGN-IN (no linkToUserId)
        # Check if GitHub UID is already linked to any user
        # ============================================================
        if is_github_flow:
            logger.info(
                f"GitHub sign-in: Checking if GitHub UID {provider_uid} is linked"
            )

            # Check if this GitHub Firebase UID is already linked
            existing_provider = (
                db.query(UserAuthProvider)
                .filter(
                    UserAuthProvider.provider_type == PROVIDER_TYPE_FIREBASE_GITHUB,
                    UserAuthProvider.provider_uid == provider_uid,
                )
                .first()
            )

            if existing_provider:
                # GitHub is linked - find the user
                user = user_service.get_user_by_uid(existing_provider.user_id)
                if user:
                    logger.info(f"GitHub {provider_uid} linked to user {user.uid}")

                    # Update last login
                    # Note: update_last_login handles encryption internally, pass plaintext token
                    if oauth_token:
                        user_service.update_last_login(user.uid, oauth_token)

                    return Response(
                        content=json.dumps(
                            {
                                "uid": user.uid,
                                "exists": True,
                                "needs_github_linking": False,
                            }
                        ),
                        status_code=200,
                    )

            # GitHub not linked - create new user with GitHub as primary
            logger.info(f"GitHub {provider_uid} not linked. Creating new user...")

            try:
                new_user, _ = await unified_auth.authenticate_or_create(
                    email=email,
                    provider_type=provider_type,  # Use the variable instead of hardcoding
                    provider_uid=provider_uid,
                    provider_data=provider_info,
                    access_token=oauth_token,
                    display_name=display_name or email.split("@")[0],
                    email_verified=email_verified,
                )

                logger.info(f"Created new user {new_user.uid} with GitHub")

                await send_slack_message(f"New signup: {email} ({display_name})")
                PostHogClient().send_event(
                    new_user.uid,
                    "signup_event",
                    {
                        "email": email,
                        "display_name": display_name,
                        "provider_type": "firebase_github",
                    },
                )

                return Response(
                    content=json.dumps(
                        {
                            "uid": new_user.uid,
                            "exists": False,
                            "needs_github_linking": False,  # They signed up with GitHub
                        }
                    ),
                    status_code=201,
                )
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to create user: {e}", exc_info=True)
                return Response(
                    content=json.dumps({"error": f"Signup failed: {str(e)}"}),
                    status_code=500,
                )

        # ============================================================
        # FLOW 3: EMAIL/PASSWORD SIGN-IN (legacy, rarely used)
        # ============================================================
        logger.info(f"Email/password flow for {email}")

        user = user_service.get_user_by_uid(uid)

        if user:
            # Existing user
            logger.info(f"Email/password user exists: {user.uid}")

            # Check GitHub linking
            has_github, _ = unified_auth.check_github_linked(user.uid)

            return Response(
                content=json.dumps(
                    {
                        "uid": user.uid,
                        "exists": True,
                        "needs_github_linking": not has_github,
                    }
                ),
                status_code=200,
            )
        else:
            # New email/password user
            try:
                new_user, _ = await unified_auth.authenticate_or_create(
                    email=email,
                    provider_type=PROVIDER_TYPE_FIREBASE_EMAIL,
                    provider_uid=uid,
                    provider_data=provider_info,
                    display_name=display_name or email.split("@")[0],
                    email_verified=email_verified,
                )

                logger.info(f"Created email/password user: {new_user.uid}")

                return Response(
                    content=json.dumps(
                        {
                            "uid": new_user.uid,
                            "exists": False,
                            "needs_github_linking": True,  # Email users always need GitHub
                        }
                    ),
                    status_code=201,
                )
            except Exception as e:
                db.rollback()
                logger.error(f"Email/password signup failed: {e}", exc_info=True)
                return Response(
                    content=json.dumps({"error": str(e)}),
                    status_code=500,
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

            # Verify the ID token with the SSO provider
            verified_user_info = await unified_auth.verify_sso_token(
                sso_request.sso_provider,
                sso_request.id_token,
            )

            if not verified_user_info:
                return JSONResponse(
                    content={"error": "Invalid or expired SSO token"},
                    status_code=401,
                )

            # Use verified data from token
            provider_data = (
                sso_request.provider_data or verified_user_info.raw_data or {}
            )
            provider_uid = (
                verified_user_info.provider_uid
                or provider_data.get("sub")
                or provider_data.get("oid")
                or sso_request.email
            )

            # Override email and display_name with verified data
            verified_email = verified_user_info.email or sso_request.email
            verified_display_name = (
                verified_user_info.display_name or provider_data.get("name")
            )

            # Authenticate or create user
            user, response = await unified_auth.authenticate_or_create(
                email=verified_email,
                provider_type=provider_type,
                provider_uid=provider_uid,
                provider_data=provider_data,
                display_name=verified_display_name or verified_email.split("@")[0],
                email_verified=verified_user_info.email_verified,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Send Slack notification for new users
            if response.status == "new_user":
                await send_slack_message(
                    f"New SSO signup: {verified_email} via {sso_request.sso_provider}"
                )

                PostHogClient().send_event(
                    user.uid,
                    "sso_signup_event",
                    {
                        "email": verified_email,
                        "sso_provider": sso_request.sso_provider,
                    },
                )

            # Note: GitHub linking check is already done in authenticate_or_create
            # for both new users (new_user status) and existing users (success status)
            # The response.needs_github_linking flag is already set correctly
            if response.status == "new_user":
                logger.info(
                    f"New user {user.uid} ({verified_email}) created via SSO. "
                    f"GitHub linking required: {response.needs_github_linking}"
                )

            return JSONResponse(
                content=response.model_dump(),
                status_code=200 if response.status == "success" else 202,
            )

        except ValueError as ve:
            logger.error(f"SSO login validation error: {str(ve)}", exc_info=True)
            return JSONResponse(
                content={"error": f"Invalid request: {str(ve)}"},
                status_code=400,
            )
        except Exception as e:
            logger.error(f"SSO login error: {str(e)}", exc_info=True)
            return JSONResponse(
                content={"error": f"SSO login failed: {str(e)}"},
                status_code=500,
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
                    "provider": AuthProviderResponse.from_orm(
                        new_provider
                    ).model_dump(),
                },
                status_code=200,
            )

        except ValueError as ve:
            logger.error(f"Provider linking validation error: {str(ve)}", exc_info=True)
            return JSONResponse(
                content={"error": f"Invalid request: {str(ve)}"},
                status_code=400,
            )
        except Exception as e:
            logger.error(f"Provider linking error: {str(e)}", exc_info=True)
            return JSONResponse(
                content={"error": f"Failed to link provider: {str(e)}"},
                status_code=500,
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
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        """
        Get all authentication providers for the current user.

        Requires authentication.
        """
        try:
            # Get user from auth token (now properly injected via Depends)
            # Firebase tokens use 'uid', but we also check 'user_id' for compatibility
            user_id = user.get("uid") or user.get("user_id")

            if not user_id:
                return JSONResponse(
                    content={"error": "Authentication required"},
                    status_code=401,
                )

            unified_auth = UnifiedAuthService(db)
            providers = unified_auth.get_user_providers(user_id)

            primary_provider = next((p for p in providers if p.is_primary), None)

            response = UserAuthProvidersResponse(
                providers=[AuthProviderResponse.from_orm(p) for p in providers],
                primary_provider=(
                    AuthProviderResponse.from_orm(primary_provider)
                    if primary_provider
                    else None
                ),
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
                (p.provider_type for p in providers if p.is_primary), None
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
