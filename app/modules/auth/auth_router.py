import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import HTTPException
from firebase_admin import auth as firebase_auth
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

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


LINK_TOKEN_TTL_SECONDS = 15 * 60  # 15 minutes


def _link_token_secret() -> str:
    """Resolve secret for HMAC-signed SSO link tokens.

    Falls back to OAUTH_STATE_SECRET to avoid yet another env var when none is
    explicitly set; if both are unset, returns "" and signing is a no-op (we
    refuse to verify in that case — see _verify_link_token).
    """
    return (
        os.getenv("SSO_LINK_TOKEN_SECRET")
        or os.getenv("OAUTH_STATE_SECRET")
        or ""
    ).strip()


def _sign_link_token(user_id: str, ttl_seconds: int = LINK_TOKEN_TTL_SECONDS) -> Optional[str]:
    """Mint a short-lived HMAC-signed proof of SSO ownership.

    Returns None if no secret is configured (caller treats absence as "feature
    disabled" rather than minting an unsigned token).
    """
    if not user_id:
        return None
    secret = _link_token_secret()
    if not secret:
        return None
    payload = {"u": user_id, "e": int(time.time()) + int(ttl_seconds)}
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8")
    sig = hmac.new(
        secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{payload_b64}.{sig}"


def _verify_link_token(token: Optional[str]) -> Optional[str]:
    """Verify a link token and return the embedded SSO user_id, else None.

    Fail-closed: if no secret is configured, verification fails (returns None)
    even if a token-shaped string is supplied. Callers must handle None.
    """
    if not token:
        return None
    secret = _link_token_secret()
    if not secret:
        # No secret → cannot verify → must fail closed
        return None
    try:
        if "." not in token:
            return None
        payload_b64, sig = token.rsplit(".", 1)
        expected = hmac.new(
            secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        )
        if int(time.time()) > int(payload.get("e", 0)):
            return None
        return payload.get("u") or None
    except Exception:
        return None


def _require_signup_auth_enabled() -> bool:
    """Hard-require an Authorization header on /signup unless explicitly disabled.

    Fail-closed default. Set POTPIE_ALLOW_UNAUTHENTICATED_SIGNUP=1 to permit the
    legacy unsafe behavior during frontend migration.
    """
    return (
        os.getenv("POTPIE_ALLOW_UNAUTHENTICATED_SIGNUP", "").strip() not in {"1", "true", "True"}
    )


def _require_link_token_enabled() -> bool:
    """Hard-require a verified linkToken when linkToUserId is set on /signup.

    Fail-closed default after the frontend has been updated. Until then ops can
    set POTPIE_ALLOW_UNVERIFIED_LINK=1 to allow the legacy unsafe behavior.
    """
    return (
        os.getenv("POTPIE_ALLOW_UNVERIFIED_LINK", "").strip() not in {"1", "true", "True"}
    )


def _verify_signup_id_token(
    request: Request, expected_uid: str, *, require_github_provider: bool
) -> Optional[dict]:
    """Verify Authorization Bearer Firebase ID token on /signup.

    Returns the decoded token on success. Raises HTTPException on failure
    unless POTPIE_ALLOW_UNAUTHENTICATED_SIGNUP=1 (legacy compat) — in which
    case returns None.
    """
    auth_header = request.headers.get("Authorization") or ""
    bearer = (
        auth_header[7:].strip()
        if auth_header.lower().startswith("bearer ")
        else None
    )

    if not bearer:
        if _require_signup_auth_enabled():
            raise HTTPException(
                status_code=401,
                detail="Signup requires an Authorization Bearer Firebase ID token.",
                headers={"WWW-Authenticate": 'Bearer realm="auth_required"'},
            )
        logger.warning(
            "Signup called without Authorization header; legacy compat allowed via "
            "POTPIE_ALLOW_UNAUTHENTICATED_SIGNUP. uid=%s",
            expected_uid,
        )
        return None

    try:
        decoded = firebase_auth.verify_id_token(bearer, check_revoked=True)
    except Exception as err:
        logger.warning("Signup: Firebase ID token verification failed: %s", err)
        raise HTTPException(
            status_code=401,
            detail="Invalid Firebase ID token",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )

    token_uid = decoded.get("uid") or decoded.get("user_id")
    if not token_uid or token_uid != expected_uid:
        logger.warning(
            "Signup: token uid (%s) does not match body uid (%s)",
            token_uid,
            expected_uid,
        )
        raise HTTPException(
            status_code=401, detail="Token does not match supplied uid"
        )

    if require_github_provider:
        sign_in_provider = (
            (decoded.get("firebase") or {}).get("sign_in_provider") or ""
        )
        if sign_in_provider != "github.com":
            logger.warning(
                "Signup GitHub flow: sign_in_provider=%s (expected github.com) uid=%s",
                sign_in_provider,
                expected_uid,
            )
            raise HTTPException(
                status_code=401,
                detail="GitHub signup flow requires a token from github.com sign-in",
            )

    return decoded


def _resolve_link_target(
    request_body: dict, *, body_link_to_user_id: Optional[str]
) -> Optional[str]:
    """Resolve the SSO user to link a new provider to, with ownership proof.

    Prefers a signed linkToken from /sso/login. Falls back to body.linkToUserId
    only when POTPIE_ALLOW_UNVERIFIED_LINK=1 (fail-closed default off).

    Returns the verified user_id, or None when no link is requested.
    Raises HTTPException(403) when verification fails.
    """
    link_token = request_body.get("linkToken") or request_body.get("link_token")
    if link_token:
        verified_uid = _verify_link_token(link_token)
        if not verified_uid:
            raise HTTPException(
                status_code=403, detail="Invalid or expired linkToken"
            )
        if (
            body_link_to_user_id
            and body_link_to_user_id != verified_uid
        ):
            logger.warning(
                "Signup link: body.linkToUserId (%s) does not match verified token uid (%s)",
                body_link_to_user_id,
                verified_uid,
            )
            raise HTTPException(
                status_code=403,
                detail="linkToken does not match linkToUserId",
            )
        return verified_uid

    if not body_link_to_user_id:
        return None

    if _require_link_token_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "linkToUserId requires a verified linkToken from /sso/login. "
                "Legacy unverified linking is disabled."
            ),
        )

    logger.warning(
        "Signup link: using unverified linkToUserId=%s (legacy compat via "
        "POTPIE_ALLOW_UNVERIFIED_LINK; vulnerable to F-3 account takeover)",
        body_link_to_user_id,
    )
    return body_link_to_user_id


def _signup_response_with_custom_token(payload: dict) -> dict:
    """No-op shim retained for compatibility.

    F-14: previously embedded a Firebase customToken in the signup response
    body. That credential is unscoped and broadens the blast radius of any
    response-body leak. Callers that genuinely need a custom token must now
    authenticate (Authorization Bearer) and call POST /auth/custom-token,
    which mints a scoped (`surface=vscode-ext`) token.
    """
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


class AuthAPI:
    @auth_router.post("/login")
    async def login(login_request: LoginRequest):
        email, password = login_request.email, login_request.password

        try:
            res = await auth_handler.login_async(email=email, password=password)
            id_token = res.get("idToken")
            return JSONResponse(content={"token": id_token}, status_code=200)
        except ValueError:
            # F-20: do not differentiate "user not found" from "wrong password";
            # both must return the same generic error to avoid email enumeration.
            return JSONResponse(
                content={"error": "Invalid email or password"}, status_code=401
            )
        except HTTPException as he:
            # F-20: static body; upstream details remain server-side only.
            logger.warning(
                "Login HTTPException upstream: status=%s", he.status_code
            )
            return JSONResponse(
                content={"error": "Invalid email or password"},
                status_code=(
                    he.status_code if 400 <= he.status_code < 500 else 401
                ),
            )
        except Exception:
            logger.exception("Login failed with unexpected error")
            return JSONResponse(
                content={"error": "Login failed"}, status_code=500
            )

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

        async_user_service = AsyncUserService(async_db)
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

        # F-3: verify Firebase ID token. For GitHub flows the body's
        # `uid` is the GitHub Firebase UID, so require decoded.uid == uid
        # AND sign_in_provider == github.com. For email/password the same uid
        # match suffices.
        # NOTE: this does NOT, on its own, close F-3 — the linkToUserId path
        # still needs ownership proof. See _resolve_link_target below.
        _verify_signup_id_token(
            request, expected_uid=uid, require_github_provider=is_github_flow
        )

        # F-3: resolve the link target via signed linkToken when available;
        # body.linkToUserId is rejected unless POTPIE_ALLOW_UNVERIFIED_LINK=1.
        try:
            link_to_user_id = _resolve_link_target(
                body, body_link_to_user_id=link_to_user_id
            )
        except HTTPException as he:
            return Response(
                content=json.dumps({"error": he.detail}),
                status_code=he.status_code,
            )

        # ============================================================
        # FLOW 1: GITHUB LINKING (linkToUserId provided)
        # User has SSO account, wants to link GitHub
        # ============================================================
        if link_to_user_id and is_github_flow:
            logger.info(f"GitHub linking: Linking GitHub to SSO user {link_to_user_id}")

            # Find the SSO user
            db.expire_all()  # Ensure fresh data
            user = await async_user_service.get_user_by_uid(link_to_user_id)

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
                        _signup_response_with_custom_token(
                            {
                                "uid": user.uid,
                                "exists": True,
                                "needs_github_linking": False,
                            }
                        )
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
                # F-17: don't echo SQLAlchemy/constraint text to the client.
                error_str = str(e).lower()
                if "unique_provider_uid" in error_str or "uniqueviolation" in error_str:
                    logger.warning(
                        "GitHub account %s already linked to another user",
                        provider_uid,
                    )
                    return Response(
                        content=json.dumps(
                            {
                                "error": (
                                    "GitHub account is already linked to another account. "
                                    "Please use a different GitHub account or contact support "
                                    "if you believe this is an error."
                                ),
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
                    _signup_response_with_custom_token(
                        {
                            "uid": user.uid,
                            "exists": True,
                            "needs_github_linking": False,
                        }
                    )
                ),
                status_code=200,
            )

        # ============================================================
        # FLOW 2: GITHUB SIGN-IN (no linkToUserId)
        # Check if GitHub UID is already linked to any user
        # BLOCK NEW GITHUB SIGNUPS
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

            # VALIDATION: Block new GitHub signups
            if not existing_provider:
                logger.warning(
                    f"Blocked new GitHub signup attempt: GitHub UID {provider_uid} is not linked to any user"
                )
                return Response(
                    content=json.dumps(
                        {
                            "error": "GitHub sign-up is no longer supported. Please use 'Continue with Google' with your work email address.",
                            "details": "New GitHub signups are disabled. Existing GitHub users can still sign in.",
                        }
                    ),
                    status_code=403,  # Forbidden
                )

            if existing_provider:
                # GitHub is linked - find the user
                user = await async_user_service.get_user_by_uid(
                    existing_provider.user_id
                )
                if user:
                    logger.info(f"GitHub {provider_uid} linked to user {user.uid}")

                    # Update last login (encrypt token before storing; update_last_login does not encrypt)
                    if oauth_token:
                        from integrations.adapters.outbound.crypto.token_encryption import (
                            encrypt_token,
                        )

                        await async_user_service.update_last_login(
                            user.uid, encrypt_token(oauth_token)
                        )

                    return Response(
                        content=json.dumps(
                            _signup_response_with_custom_token(
                                {
                                    "uid": user.uid,
                                    "exists": True,
                                    "needs_github_linking": False,
                                }
                            )
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

                background_tasks.add_task(
                    send_slack_message, f"New signup: {email} ({display_name})"
                )
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
                        _signup_response_with_custom_token(
                            {
                                "uid": new_user.uid,
                                "exists": False,
                                "needs_github_linking": False,  # They signed up with GitHub
                            }
                        )
                    ),
                    status_code=201,
                )
            except RuntimeError as ue:
                # F-5 fail-closed: Firebase identity provider unavailable
                db.rollback()
                logger.error("Signup blocked: %s", ue)
                return Response(
                    content=json.dumps(
                        {"error": "Identity provider unavailable"}
                    ),
                    status_code=503,
                )
            except Exception:
                db.rollback()
                logger.exception("Failed to create GitHub user")
                return Response(
                    content=json.dumps({"error": "Signup failed"}),
                    status_code=500,
                )

        # ============================================================
        # FLOW 3: EMAIL/PASSWORD SIGN-IN (legacy, rarely used)
        # ============================================================
        logger.info(f"Email/password flow for {email}")

        user = await async_user_service.get_user_by_uid(uid)

        if user:
            # Existing user
            logger.info(f"Email/password user exists: {user.uid}")

            # Check GitHub linking
            has_github, _ = unified_auth.check_github_linked(user.uid)

            return Response(
                content=json.dumps(
                    _signup_response_with_custom_token(
                        {
                            "uid": user.uid,
                            "exists": True,
                            "needs_github_linking": not has_github,
                        }
                    )
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
                background_tasks.add_task(
                    send_slack_message, f"New signup: {email} ({display_name})"
                )

                return Response(
                    content=json.dumps(
                        _signup_response_with_custom_token(
                            {
                                "uid": new_user.uid,
                                "exists": False,
                                "needs_github_linking": True,  # Email users always need GitHub
                            }
                        )
                    ),
                    status_code=201,
                )
            except RuntimeError as ue:
                # F-5 fail-closed: Firebase identity provider unavailable
                db.rollback()
                logger.error("Signup blocked: %s", ue)
                return Response(
                    content=json.dumps(
                        {"error": "Identity provider unavailable"}
                    ),
                    status_code=503,
                )
            except Exception:
                db.rollback()
                logger.exception("Email/password signup failed")
                return Response(
                    content=json.dumps({"error": "Signup failed"}),
                    status_code=500,
                )

    @auth_router.post("/auth/custom-token")
    async def custom_token(user=Depends(AuthService.check_auth)):
        """
        Create a scoped Firebase custom token for the authenticated user
        (e.g. for VS Code extension). Requires Authorization Bearer.

        F-14: token carries `surface=vscode-ext` so a leaked custom token
        doesn't grant general API access.
        """
        uid = user.get("uid") or user.get("user_id")
        if not uid:
            return Response(
                content=json.dumps({"error": "Missing uid in token"}),
                status_code=401,
            )
        custom_token = AuthService.create_custom_token(
            uid, additional_claims={"surface": "vscode-ext"}
        )
        if not custom_token:
            return Response(
                content=json.dumps({"error": "Failed to create custom token"}),
                status_code=500,
            )
        return Response(
            content=json.dumps({"customToken": custom_token}),
            status_code=200,
        )

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

            # F-3: mint server-signed proof of SSO ownership so the frontend
            # can forward it to /signup as `linkToken` instead of relying on
            # body-trusted linkToUserId.
            if user and user.uid:
                response.link_token = _sign_link_token(user.uid)

            return JSONResponse(
                content=response.model_dump(),
                status_code=200 if response.status == "success" else 202,
            )

        except ValueError:
            logger.exception("SSO login validation error")
            return JSONResponse(
                content={"error": "Invalid request"},
                status_code=400,
            )
        except RuntimeError as ue:
            # F-5 fail-closed: Firebase identity provider unavailable
            logger.error("SSO login blocked: %s", ue)
            return JSONResponse(
                content={"error": "Identity provider unavailable"},
                status_code=503,
            )
        except Exception:
            logger.exception("SSO login error")
            return JSONResponse(
                content={"error": "SSO login failed"},
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
                    "provider": AuthProviderResponse.from_orm(new_provider).model_dump(
                        mode="json"
                    ),
                },
                status_code=200,
            )

        except ValueError:
            logger.exception("Provider linking validation error")
            return JSONResponse(
                content={"error": "Invalid request"},
                status_code=400,
            )
        except Exception:
            logger.exception("Provider linking error")
            return JSONResponse(
                content={"error": "Failed to link provider"},
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

        except Exception:
            logger.exception("Failed to cancel linking")
            return JSONResponse(
                content={"error": "Failed to cancel linking"},
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
                content=response.model_dump(mode="json"),
                status_code=200,
            )

        except Exception:
            logger.exception("Failed to get providers")
            return JSONResponse(
                content={"error": "Failed to get providers"},
                status_code=400,
            )

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

        except Exception:
            logger.exception("Failed to set primary provider")
            return JSONResponse(
                content={"error": "Failed to set primary provider"},
                status_code=400,
            )

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

            except ValueError as ve:
                # Cannot unlink last provider
                return JSONResponse(
                    content={"error": str(ve)},
                    status_code=400,
                )

        except Exception:
            logger.exception("Failed to unlink provider")
            return JSONResponse(
                content={"error": "Failed to unlink provider"},
                status_code=400,
            )

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

        except Exception:
            logger.exception("Failed to get account")
            return JSONResponse(
                content={"error": "Failed to get account"},
                status_code=400,
            )
