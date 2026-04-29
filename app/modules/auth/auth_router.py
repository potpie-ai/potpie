import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_db, get_db
from app.modules.auth.auth_schema import (
    LoginRequest,
    SSOLoginRequest,
    ConfirmLinkingRequest,
    UnlinkProviderRequest,
    SetPrimaryProviderRequest,
    UserAuthProvidersResponse,
    AuthProviderResponse,
    AccountResponse,
    AuthProviderCreate,
)
from app.modules.auth.auth_service import auth_handler, AuthService
from app.modules.auth.auth_provider_model import UserAuthProvider
from app.modules.auth.unified_auth_service import (
    UnifiedAuthService,
    PROVIDER_TYPE_FIREBASE_GITHUB,
    PROVIDER_TYPE_FIREBASE_EMAIL,
)
from app.modules.users.user_service import AsyncUserService
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.posthog_helper import PostHogClient
from app.modules.utils.email_helper import is_personal_email_domain

logger = logging.getLogger(__name__)

auth_router = APIRouter()
load_dotenv(override=True)

GITHUB_LINK_CONFLICT_ERROR = (
    "GitHub account is already linked to another account. "
    "Please use a different GitHub account or contact support if you believe this is an error."
)
GITHUB_SIGNUP_DISABLED_ERROR = (
    "GitHub sign-up is no longer supported. Please use 'Continue with Google' "
    "with your work email address."
)
GITHUB_SIGNUP_DISABLED_DETAILS = (
    "New GitHub signups are disabled. Existing GitHub users can still sign in."
)
GENERIC_SIGNUP_FAILURE_ERROR = "Signup failed"
GENERIC_REQUEST_FAILURE_ERROR = "Unable to process request"
INVALID_REQUEST_ERROR = "Invalid request"


@dataclass(frozen=True)
class SignupRequestData:
    uid: str | None
    email: str | None
    display_name: str | None
    email_verified: bool
    link_to_user_id: str | None
    github_firebase_uid: str | None
    oauth_token: str | None
    provider_username: str | None
    provider_info: dict[str, Any]

    @property
    def is_github_flow(self) -> bool:
        return bool(self.oauth_token and self.provider_username)

    @property
    def provider_type(self) -> str:
        return (
            PROVIDER_TYPE_FIREBASE_GITHUB
            if self.is_github_flow
            else PROVIDER_TYPE_FIREBASE_EMAIL
        )

    @property
    def provider_uid(self) -> str | None:
        return self.github_firebase_uid or self.uid


def _signup_response_with_custom_token(payload: dict) -> dict:
    """Add customToken to signup payload when applicable (e.g. for VS Code extension)."""
    uid = payload.get("uid")
    if uid:
        custom_token = AuthService.create_custom_token(uid)
        if custom_token:
            payload["customToken"] = custom_token
    return payload


def _get_slack_webhook_url() -> str | None:
    """Resolve Slack webhook URL at call time so dotenv/env changes are respected."""
    return os.getenv("SLACK_WEBHOOK_URL")


async def send_slack_message(message: str):
    payload = {"text": message}
    webhook_url = _get_slack_webhook_url()
    if webhook_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(webhook_url, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Slack webhook returned non-2xx status: %s body=%s",
                e.response.status_code,
                e.response.text,
            )
        except Exception as e:
            logger.warning("Failed to send Slack signup notification: %s", e)


def _mask_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return email
    local_part, domain = email.split("@", 1)
    if not local_part:
        return f"***@{domain}"
    return f"{local_part[0]}***@{domain}"


def _json_error_response(
    error: str, status_code: int, details: str | None = None
) -> JSONResponse:
    content = {"error": error}
    if details:
        content["details"] = details
    return JSONResponse(content=content, status_code=status_code)


def _signup_success_response(
    uid: str, exists: bool, needs_github_linking: bool, status_code: int
) -> JSONResponse:
    return JSONResponse(
        content=_signup_response_with_custom_token(
            {
                "uid": uid,
                "exists": exists,
                "needs_github_linking": needs_github_linking,
            }
        ),
        status_code=status_code,
    )


def _extract_signup_provider_info(
    body: dict[str, Any], provider_username: str | None
) -> dict[str, Any]:
    provider_info: dict[str, Any] = {}
    provider_data = body.get("providerData")

    if isinstance(provider_data, list) and provider_data:
        provider_info = (
            provider_data[0].copy() if isinstance(provider_data[0], dict) else {}
        )
    elif isinstance(provider_data, dict):
        provider_info = provider_data.copy()

    if provider_username:
        provider_info["username"] = provider_username

    return provider_info


def _parse_signup_request(body: dict[str, Any]) -> SignupRequestData:
    provider_username = body.get("providerUsername")
    return SignupRequestData(
        uid=body.get("uid"),
        email=body.get("email"),
        display_name=body.get("displayName") or body.get("display_name"),
        email_verified=body.get("emailVerified") or body.get("email_verified", False),
        link_to_user_id=body.get("linkToUserId"),
        github_firebase_uid=body.get("githubFirebaseUid"),
        oauth_token=body.get("accessToken") or body.get("access_token"),
        provider_username=provider_username,
        provider_info=_extract_signup_provider_info(body, provider_username),
    )


def _log_signup_request(signup_data: SignupRequestData) -> None:
    logger.info(
        "Signup: uid=%s, email=%s, linkToUserId=%s, githubFirebaseUid=%s, hasToken=%s",
        signup_data.uid,
        _mask_email(signup_data.email),
        signup_data.link_to_user_id,
        signup_data.github_firebase_uid,
        bool(signup_data.oauth_token),
    )


def _validate_signup_request(signup_data: SignupRequestData) -> JSONResponse | None:
    if not signup_data.uid:
        return _json_error_response("Missing uid", 400)
    if not signup_data.email:
        return _json_error_response("Missing email", 400)
    return None


def _get_existing_github_provider(db: Session, user_id: str) -> UserAuthProvider | None:
    return (
        db.query(UserAuthProvider)
        .filter(
            UserAuthProvider.user_id == user_id,
            UserAuthProvider.provider_type == PROVIDER_TYPE_FIREBASE_GITHUB,
        )
        .first()
    )


def _get_github_provider_by_uid(
    db: Session, provider_uid: str | None
) -> UserAuthProvider | None:
    return (
        db.query(UserAuthProvider)
        .filter(
            UserAuthProvider.provider_type == PROVIDER_TYPE_FIREBASE_GITHUB,
            UserAuthProvider.provider_uid == provider_uid,
        )
        .first()
    )


def _build_github_provider_create(signup_data: SignupRequestData) -> AuthProviderCreate:
    return AuthProviderCreate(
        provider_type=signup_data.provider_type,
        provider_uid=signup_data.provider_uid,
        provider_data=signup_data.provider_info,
        access_token=signup_data.oauth_token,
        is_primary=False,
    )


def _queue_signup_notification(
    background_tasks: BackgroundTasks, email: str, display_name: str | None
) -> None:
    background_tasks.add_task(
        send_slack_message, f"New signup: {email} ({display_name})"
    )


def _default_display_name(signup_data: SignupRequestData) -> str:
    return signup_data.display_name or signup_data.email.split("@")[0]


def _is_provider_uid_conflict(error: IntegrityError) -> bool:
    error_str = str(error).lower()
    return "unique_provider_uid" in error_str or "uniqueviolation" in error_str


def _link_github_provider(
    unified_auth: UnifiedAuthService,
    db: Session,
    user_id: str,
    signup_data: SignupRequestData,
) -> JSONResponse | None:
    try:
        unified_auth.add_provider(
            user_id=user_id,
            provider_create=_build_github_provider_create(signup_data),
        )
        db.commit()
        return None
    except IntegrityError as error:
        db.rollback()
        if _is_provider_uid_conflict(error):
            logger.error(
                "GitHub account %s is already linked to another user: %s",
                signup_data.provider_uid,
                error,
            )
            return _json_error_response(GITHUB_LINK_CONFLICT_ERROR, 409)
        raise
    except Exception as error:
        db.rollback()
        logger.error(
            "Unexpected error linking GitHub provider: %s", error, exc_info=True
        )
        raise


async def _update_github_last_login(
    async_user_service: AsyncUserService,
    user_id: str,
    oauth_token: str | None,
) -> None:
    if not oauth_token:
        return

    from app.modules.integrations.token_encryption import encrypt_token

    await async_user_service.update_last_login(user_id, encrypt_token(oauth_token))


async def _handle_github_linking_signup(
    signup_data: SignupRequestData,
    async_user_service: AsyncUserService,
    unified_auth: UnifiedAuthService,
    db: Session,
) -> JSONResponse:
    logger.info(
        "GitHub linking: Linking GitHub to SSO user %s", signup_data.link_to_user_id
    )

    db.expire_all()
    user = await async_user_service.get_user_by_uid(signup_data.link_to_user_id)
    if not user:
        logger.error("SSO user %s not found in database!", signup_data.link_to_user_id)
        return _json_error_response("User not found. Please sign in again.", 404)

    logger.info("Found SSO user: uid=%s, email=%s", user.uid, user.email)

    if _get_existing_github_provider(db, user.uid):
        logger.info("GitHub already linked to user %s", user.uid)
        return _signup_success_response(
            user.uid, exists=True, needs_github_linking=False, status_code=200
        )

    existing_provider_with_uid = _get_github_provider_by_uid(
        db, signup_data.provider_uid
    )
    if existing_provider_with_uid and existing_provider_with_uid.user_id != user.uid:
        logger.warning(
            "GitHub account %s is already linked to user %s, cannot link to user %s",
            signup_data.provider_uid,
            existing_provider_with_uid.user_id,
            user.uid,
        )
        return _json_error_response(GITHUB_LINK_CONFLICT_ERROR, 409)

    logger.info(
        "Linking GitHub (provider_uid=%s) to user %s",
        signup_data.provider_uid,
        user.uid,
    )
    link_response = _link_github_provider(unified_auth, db, user.uid, signup_data)
    if link_response:
        return link_response

    logger.info("Successfully linked GitHub to user %s", user.uid)
    return _signup_success_response(
        user.uid, exists=True, needs_github_linking=False, status_code=200
    )


async def _create_github_signup_user(
    signup_data: SignupRequestData,
    unified_auth: UnifiedAuthService,
    background_tasks: BackgroundTasks,
    db: Session,
) -> JSONResponse:
    try:
        new_user, _ = await unified_auth.authenticate_or_create(
            email=signup_data.email,
            provider_type=signup_data.provider_type,
            provider_uid=signup_data.provider_uid,
            provider_data=signup_data.provider_info,
            access_token=signup_data.oauth_token,
            display_name=_default_display_name(signup_data),
            email_verified=signup_data.email_verified,
        )

        logger.info("Created new user %s with GitHub", new_user.uid)
        _queue_signup_notification(
            background_tasks, signup_data.email, signup_data.display_name
        )
        PostHogClient().send_event(
            new_user.uid,
            "signup_event",
            {
                "email": signup_data.email,
                "display_name": signup_data.display_name,
                "provider_type": "firebase_github",
            },
        )
        return _signup_success_response(
            new_user.uid, exists=False, needs_github_linking=False, status_code=201
        )
    except Exception as error:
        db.rollback()
        logger.error("Failed to create user: %s", error, exc_info=True)
        return _json_error_response(GENERIC_SIGNUP_FAILURE_ERROR, 500)


async def _handle_github_signin_signup(
    signup_data: SignupRequestData,
    async_user_service: AsyncUserService,
    unified_auth: UnifiedAuthService,
    background_tasks: BackgroundTasks,
    db: Session,
) -> JSONResponse:
    logger.info(
        "GitHub sign-in: Checking if GitHub UID %s is linked",
        signup_data.provider_uid,
    )

    existing_provider = _get_github_provider_by_uid(db, signup_data.provider_uid)
    if not existing_provider:
        logger.warning(
            "Blocked new GitHub signup attempt: GitHub UID %s is not linked to any user",
            signup_data.provider_uid,
        )
        return _json_error_response(
            GITHUB_SIGNUP_DISABLED_ERROR,
            403,
            details=GITHUB_SIGNUP_DISABLED_DETAILS,
        )

    user = await async_user_service.get_user_by_uid(existing_provider.user_id)
    if user:
        logger.info("GitHub %s linked to user %s", signup_data.provider_uid, user.uid)
        await _update_github_last_login(
            async_user_service, user.uid, signup_data.oauth_token
        )
        return _signup_success_response(
            user.uid, exists=True, needs_github_linking=False, status_code=200
        )

    logger.info("GitHub %s not linked. Creating new user...", signup_data.provider_uid)
    return await _create_github_signup_user(
        signup_data, unified_auth, background_tasks, db
    )


async def _handle_email_password_signup(
    signup_data: SignupRequestData,
    async_user_service: AsyncUserService,
    unified_auth: UnifiedAuthService,
    background_tasks: BackgroundTasks,
    db: Session,
) -> JSONResponse:
    logger.info("Email/password flow for %s", signup_data.email)
    user = await async_user_service.get_user_by_uid(signup_data.uid)

    if user:
        logger.info("Email/password user exists: %s", user.uid)
        has_github, _ = unified_auth.check_github_linked(user.uid)
        return _signup_success_response(
            user.uid,
            exists=True,
            needs_github_linking=not has_github,
            status_code=200,
        )

    try:
        new_user, _ = await unified_auth.authenticate_or_create(
            email=signup_data.email,
            provider_type=PROVIDER_TYPE_FIREBASE_EMAIL,
            provider_uid=signup_data.uid,
            provider_data=signup_data.provider_info,
            display_name=_default_display_name(signup_data),
            email_verified=signup_data.email_verified,
        )

        logger.info("Created email/password user: %s", new_user.uid)
        _queue_signup_notification(
            background_tasks, signup_data.email, signup_data.display_name
        )
        return _signup_success_response(
            new_user.uid, exists=False, needs_github_linking=True, status_code=201
        )
    except Exception as error:
        db.rollback()
        logger.error("Email/password signup failed: %s", error, exc_info=True)
        return _json_error_response(GENERIC_SIGNUP_FAILURE_ERROR, 500)


class AuthAPI:
    @auth_router.post("/login")
    async def login(login_request: LoginRequest):
        email, password = login_request.email, login_request.password

        try:
            res = await auth_handler.login_async(email=email, password=password)
            id_token = res.get("idToken")
            return JSONResponse(content={"token": id_token}, status_code=200)
        except ValueError:
            return JSONResponse(
                content={"error": "Invalid email or password"}, status_code=401
            )
        except HTTPException as he:
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, he.status_code)
        except Exception as e:
            logger.error("Login error: %s", e, exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 400)

    @staticmethod
    @auth_router.post("/signup")
    async def signup(
        request: Request,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
    ):
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
        signup_data = _parse_signup_request(await request.json())
        _log_signup_request(signup_data)

        validation_error = _validate_signup_request(signup_data)
        if validation_error:
            return validation_error

        async_user_service = AsyncUserService(async_db)
        unified_auth = UnifiedAuthService(db)

        if signup_data.link_to_user_id and signup_data.is_github_flow:
            return await _handle_github_linking_signup(
                signup_data, async_user_service, unified_auth, db
            )

        if signup_data.is_github_flow:
            return await _handle_github_signin_signup(
                signup_data,
                async_user_service,
                unified_auth,
                background_tasks,
                db,
            )

        return await _handle_email_password_signup(
            signup_data,
            async_user_service,
            unified_auth,
            background_tasks,
            db,
        )

    @auth_router.post("/auth/custom-token")
    async def custom_token(user=Depends(AuthService.check_auth)):
        """
        Create a Firebase custom token for the authenticated user (e.g. for VS Code extension).
        Requires Authorization: Bearer <firebase_id_token>.
        Returns { "customToken": "..." }.
        """
        uid = user.get("uid") or user.get("user_id")
        if not uid:
            return JSONResponse(
                content={"error": "Missing uid in token"},
                status_code=401,
            )
        custom_token = AuthService.create_custom_token(uid)
        if not custom_token:
            return JSONResponse(
                content={"error": "Failed to create custom token"},
                status_code=500,
            )
        return JSONResponse(content={"customToken": custom_token}, status_code=200)

    # ===== Multi-Provider SSO Endpoints =====

    @auth_router.post("/sso/login")
    async def sso_login(
        request: Request,
        sso_request: SSOLoginRequest,
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
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

            # VALIDATION: Block new users with generic emails (for Google SSO)
            # Check if user already exists by email (legacy user check)
            async_user_service = AsyncUserService(async_db)
            existing_user = await async_user_service.get_user_by_email(verified_email)

            if is_personal_email_domain(verified_email):
                if not existing_user:
                    # New user with generic email - block them
                    logger.warning(
                        f"Blocked new signup attempt with generic email: {verified_email} via {sso_request.sso_provider}"
                    )
                    return JSONResponse(
                        content={
                            "error": "Personal email addresses are not allowed. Please use your work/corporate email to sign in.",
                            "details": "Generic email providers (Gmail, Yahoo, Outlook, etc.) cannot be used for new signups.",
                        },
                        status_code=403,  # Forbidden
                    )
                # Existing user with generic email - allow (legacy policy)
                logger.info(
                    f"Allowing legacy user with generic email: {verified_email}"
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
            return _json_error_response(INVALID_REQUEST_ERROR, 400)
        except Exception as e:
            logger.error(f"SSO login error: {str(e)}", exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 500)

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
                    "provider": AuthProviderResponse.from_orm(new_provider).model_dump(
                        mode="json"
                    ),
                },
                status_code=200,
            )

        except ValueError as ve:
            logger.error(f"Provider linking validation error: {str(ve)}", exc_info=True)
            return _json_error_response(INVALID_REQUEST_ERROR, 400)
        except Exception as e:
            logger.error(f"Provider linking error: {str(e)}", exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 500)

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
            logger.error("Cancel provider linking error: %s", e, exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 400)

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
                content=response.model_dump(mode="json"),
                status_code=200,
            )

        except Exception as e:
            logger.error("Get providers error: %s", e, exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 400)

    @auth_router.post("/providers/set-primary")
    async def set_primary_provider(
        primary_request: SetPrimaryProviderRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        """Set a provider as the primary login method"""
        try:
            user_id = user.get("uid") or user.get("user_id")

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
            logger.error("Set primary provider error: %s", e, exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 400)

    @auth_router.delete("/providers/unlink")
    async def unlink_provider(
        unlink_request: UnlinkProviderRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        """Unlink a provider from account"""
        try:
            user_id = user.get("uid") or user.get("user_id")

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

            except ValueError:
                # Cannot unlink last provider
                return _json_error_response(INVALID_REQUEST_ERROR, 400)

        except Exception as e:
            logger.error("Unlink provider error: %s", e, exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 400)

    @auth_router.get("/account/me")
    async def get_my_account(
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
        async_db: AsyncSession = Depends(get_async_db),
    ):
        """Get complete account information including all providers"""
        try:
            user_id = user.get("uid") or user.get("user_id")

            if not user_id:
                return JSONResponse(
                    content={"error": "Authentication required"},
                    status_code=401,
                )

            async_user_service = AsyncUserService(async_db)
            user = await async_user_service.get_user_by_uid(user_id)

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
                content=response.model_dump(mode="json"),
                status_code=200,
            )

        except Exception as e:
            logger.error("Get account error: %s", e, exc_info=True)
            return _json_error_response(GENERIC_REQUEST_FAILURE_ERROR, 400)
