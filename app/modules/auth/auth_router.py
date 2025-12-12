import json
import logging
import os
import ssl
import traceback
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sqlalchemy.orm import Session
from typing import Optional, Tuple, Dict, Any

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
    AuthProviderCreate,
)
from app.modules.auth.auth_service import auth_handler
from app.modules.auth.unified_auth_service import UnifiedAuthService
from app.modules.users.user_schema import CreateUser
from app.modules.users.user_service import UserService
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.posthog_helper import PostHogClient

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", None)
AUTH_REQUIRED_ERROR = "Authentication required"

auth_router = APIRouter()
load_dotenv(override=True)


async def send_slack_message(message: str):
    payload = {"text": message}
    if SLACK_WEBHOOK_URL:
        # Use secure TLS defaults with explicit minimum protocol version
        ssl_context = ssl.create_default_context()
        # Explicitly set minimum TLS version for security
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        async with httpx.AsyncClient(verify=ssl_context) as client:
            try:
                await client.post(SLACK_WEBHOOK_URL, json=payload, timeout=10.0)
            except Exception as e:
                logger.warning(f"Failed to send Slack message: {str(e)}")


class AuthAPI:
    @staticmethod
    async def _link_accounts_by_email(
        db: Session, user_by_email, uid: str, email: str
    ) -> Response:
        """Link accounts when user exists by email but not by UID."""
        logger.info(
            f"Email {email} exists with different UID. Linking accounts: {user_by_email.uid} -> {uid}"
        )
        existing_uid = user_by_email.uid

        try:
            user_by_email.uid = uid
            db.commit()
            logger.info(
                f"Updated user UID from {existing_uid} to {uid} for email {email}"
            )

            unified_auth = UnifiedAuthService(db)
            provider_create = AuthProviderCreate(
                provider_type="firebase_email",
                provider_uid=uid,
                provider_data={"email": email, "uid": uid},
                access_token=None,
                is_primary=False,
            )
            unified_auth.add_provider(uid, provider_create)
            logger.info(f"Added firebase_email provider for user {uid}")

            return Response(
                content=json.dumps({"uid": uid, "exists": True, "linked": True}),
                status_code=200,
            )
        except Exception as e:
            logger.error(f"Error linking accounts: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            db.rollback()
            return Response(
                content=json.dumps({"error": f"Failed to link accounts: {str(e)}"}),
                status_code=400,
            )

    @staticmethod
    def _handle_existing_user_github(
        db: Session,
        uid: str,
        oauth_token: str,
        provider_data: list,
        provider_username: str,
    ) -> None:
        """Handle GitHub provider for existing user."""
        try:
            unified_auth = UnifiedAuthService(db)
            provider_info = (
                provider_data[0] if isinstance(provider_data, list) else provider_data
            )

            existing_github = unified_auth.get_provider(uid, "firebase_github")
            if not existing_github:
                provider_create = AuthProviderCreate(
                    provider_type="firebase_github",
                    provider_uid=provider_username or uid,
                    provider_data=provider_info,
                    access_token=oauth_token,
                    is_primary=False,
                )
                unified_auth.add_provider(uid, provider_create)
                logger.info(f"Linked GitHub provider to existing user {uid}")
            else:
                all_providers = unified_auth.get_user_providers(uid)
                has_other_providers = any(
                    p.provider_type != "firebase_github" for p in all_providers
                )

                if not existing_github.is_primary and not has_other_providers:
                    logger.info(
                        f"User {uid} signed in with GitHub (only provider) - "
                        "setting GitHub as primary provider"
                    )
                    unified_auth.set_primary_provider(uid, "firebase_github")
                elif not existing_github.is_primary and has_other_providers:
                    logger.info(
                        f"User {uid} linking GitHub (has other providers) - "
                        "keeping existing primary provider"
                    )
                unified_auth.update_last_used(uid, "firebase_github")
        except Exception as e:
            logger.warning(f"Failed to add GitHub provider for existing user: {str(e)}")

    @staticmethod
    def _handle_existing_user_email(db: Session, uid: str, email: str) -> None:
        """Handle email provider for existing user."""
        try:
            unified_auth = UnifiedAuthService(db)
            existing_email_provider = unified_auth.get_provider(uid, "firebase_email")
            if not existing_email_provider:
                provider_create = AuthProviderCreate(
                    provider_type="firebase_email",
                    provider_uid=uid,
                    provider_data={"email": email, "uid": uid},
                    access_token=None,
                    is_primary=False,
                )
                unified_auth.add_provider(uid, provider_create)
                logger.info(f"Added firebase_email provider for existing user {uid}")
        except Exception as e:
            logger.warning(f"Failed to add email provider for existing user: {str(e)}")

    @staticmethod
    def _link_after_duplicate_error(
        db: Session, uid: str, email: str
    ) -> Optional[Response]:
        """Link accounts after duplicate email error during user creation."""
        user_service = UserService(db)
        user_by_email = user_service.get_user_by_email(email)
        if not user_by_email:
            return None

        logger.info(
            f"User creation failed due to duplicate email. Linking {uid} to existing user {user_by_email.uid}"
        )
        try:
            user_by_email.uid = uid
            db.commit()

            unified_auth = UnifiedAuthService(db)
            provider_create = AuthProviderCreate(
                provider_type="firebase_email",
                provider_uid=uid,
                provider_data={"email": email, "uid": uid},
                access_token=None,
                is_primary=False,
            )
            unified_auth.add_provider(uid, provider_create)

            return Response(
                content=json.dumps({"uid": uid, "exists": True, "linked": True}),
                status_code=200,
            )
        except Exception as e:
            logger.error(
                f"Error linking accounts after duplicate email error: {str(e)}"
            )
            db.rollback()
            return None

    @staticmethod
    def _create_provider_info(
        provider_data: list, oauth_token: str, uid: str, email: str
    ) -> Dict[str, Any]:
        """Create provider info dict for user creation."""
        if provider_data and len(provider_data) > 0:
            provider_info = (
                provider_data[0] if isinstance(provider_data, list) else provider_data
            )
            if oauth_token:
                provider_info["access_token"] = oauth_token
            return provider_info
        return {"providerId": "password", "uid": uid, "email": email}

    @staticmethod
    def _determine_user_email(
        user_service: UserService, email: str, oauth_token: str, provider_data: list
    ) -> str:
        """Determine the correct email to use for new user creation."""
        if oauth_token and provider_data and len(provider_data) > 0:
            existing_user_by_email = user_service.get_user_by_email(email)
            if existing_user_by_email:
                logger.info(
                    f"GitHub signup for existing email {email}. "
                    f"Using existing user email: {existing_user_by_email.email}"
                )
                return existing_user_by_email.email
        return email

    @staticmethod
    async def _verify_sso_token_and_extract_info(
        unified_auth: UnifiedAuthService,
        sso_provider: str,
        id_token: str,
        request_email: str,
        provider_data: dict,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
        """Verify SSO token and extract user info. Returns (user_info_dict, error_response)."""
        try:
            user_info = await unified_auth.verify_sso_token(sso_provider, id_token)
            if not user_info:
                logger.error(f"Token verification failed for {sso_provider}")
                return None, JSONResponse(
                    content={"error": "Invalid or expired ID token"},
                    status_code=401,
                )

            verified_email = user_info.email
            provider_uid = user_info.provider_uid
            display_name = user_info.display_name or verified_email.split("@")[0]
            email_verified = user_info.email_verified

            # Security: Verify that the email in the request matches the verified token email
            if request_email.lower() != verified_email.lower():
                logger.warning(
                    f"Email mismatch: request email {request_email} does not match "
                    f"verified token email {verified_email}"
                )
                return None, JSONResponse(
                    content={
                        "error": "Email in request does not match verified token email"
                    },
                    status_code=400,
                )

            # Build provider_data from verified token
            provider_data["email"] = verified_email
            if user_info.raw_data:
                provider_data.update(user_info.raw_data)

            return {
                "email": verified_email,
                "provider_uid": provider_uid,
                "display_name": display_name,
                "email_verified": email_verified,
                "provider_data": provider_data,
            }, None

        except ValueError as e:
            logger.error(f"Token verification error: {str(e)}")
            return None, JSONResponse(
                content={"error": f"Token verification failed: {str(e)}"},
                status_code=401,
            )
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {str(e)}", exc_info=True)
            return None, JSONResponse(
                content={"error": "Token verification failed"},
                status_code=500,
            )

    @staticmethod
    def _create_firebase_token(user: any) -> Optional[str]:
        """Create Firebase custom token for user. Returns token string or None."""
        if os.getenv("isDevelopmentMode") == "enabled":
            logger.info(
                "Development mode enabled - skipping Firebase custom token creation"
            )
            return None

        try:
            import firebase_admin
            from firebase_admin import auth as firebase_auth

            # Check if Firebase is initialized
            try:
                firebase_admin.get_app()
                logger.info("Firebase Admin is initialized")
            except ValueError:
                logger.error(
                    "Firebase Admin not initialized. Cannot create custom token."
                )
                logger.error("Make sure Firebase is initialized in app startup")
                raise RuntimeError("Firebase Admin not initialized")

            # Create or get Firebase user
            try:
                firebase_user = firebase_auth.get_user_by_email(user.email)
                token_uid = (
                    firebase_user.uid if firebase_user.uid != user.uid else user.uid
                )
                if firebase_user.uid != user.uid:
                    logger.warning(
                        f"Firebase UID mismatch: {firebase_user.uid} != {user.uid}"
                    )
            except firebase_auth.UserNotFoundError:
                # Create Firebase user if it doesn't exist
                try:
                    firebase_auth.create_user(
                        uid=user.uid,
                        email=user.email,
                        display_name=user.display_name,
                        email_verified=True,
                    )
                    token_uid = user.uid
                    logger.info(
                        f"Created Firebase user {user.uid} for email {user.email}"
                    )
                except firebase_auth.UidAlreadyExistsError:
                    firebase_user = firebase_auth.get_user(user.uid)
                    token_uid = user.uid

            # Generate custom token
            custom_token = firebase_auth.create_custom_token(token_uid)
            return custom_token.decode("utf-8")

        except ImportError as e:
            logger.error(f"Firebase Admin not available: {str(e)}")
            logger.error(
                "Cannot create custom token - Firebase Admin SDK not installed"
            )
            return None
        except ValueError as e:
            logger.error(f"Firebase Admin not initialized: {str(e)}")
            logger.error("Cannot create custom token - Firebase not initialized")
            return None
        except Exception as firebase_error:
            logger.error(
                f"Failed to create Firebase custom token: {str(firebase_error)}"
            )
            logger.error(f"Error type: {type(firebase_error).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    @staticmethod
    def _determine_display_email(
        user: any, providers: list, primary_provider_obj: Optional[any]
    ) -> str:
        """Determine the correct email to return based on primary provider and fallback logic."""
        display_email = user.email  # Default to database email

        if primary_provider_obj:
            # Get email from primary provider's data
            if primary_provider_obj.provider_data:
                provider_email = primary_provider_obj.provider_data.get("email")
                if provider_email:
                    return provider_email
        else:
            # No primary provider - try to find best email
            # Priority: SSO > Email/Password > GitHub > Database
            display_email = AuthAPI._get_email_from_sso_provider(providers, user.email)
            if display_email != user.email:
                return display_email

            display_email = AuthAPI._get_email_from_email_provider(
                providers, user.email
            )
            if display_email != user.email:
                return display_email

            display_email = AuthAPI._get_email_from_github_provider(
                providers, user.email
            )

        return display_email

    @staticmethod
    def _get_email_from_sso_provider(providers: list, default_email: str) -> str:
        """Get email from SSO provider if available."""
        sso_provider = next(
            (p for p in providers if p.provider_type.startswith("sso_")), None
        )
        if sso_provider and sso_provider.provider_data:
            provider_email = sso_provider.provider_data.get("email")
            if provider_email:
                return provider_email
        return default_email

    @staticmethod
    def _get_email_from_email_provider(providers: list, default_email: str) -> str:
        """Get email from email/password provider if available."""
        email_provider = next(
            (p for p in providers if p.provider_type == "firebase_email"), None
        )
        if email_provider and email_provider.provider_data:
            provider_email = email_provider.provider_data.get("email")
            if provider_email:
                return provider_email
        return default_email

    @staticmethod
    def _get_email_from_github_provider(providers: list, default_email: str) -> str:
        """Get email from GitHub provider if available."""
        github_provider = next(
            (p for p in providers if p.provider_type == "firebase_github"), None
        )
        if github_provider and github_provider.provider_data:
            provider_email = github_provider.provider_data.get("email") or (
                github_provider.provider_data.get("login")
            )
            if provider_email:
                return provider_email
        return default_email

    @staticmethod
    def _add_provider_for_new_user(
        db: Session,
        uid: str,
        email: str,
        oauth_token: str,
        provider_data: list,
        provider_username: str,
    ) -> None:
        """Add provider for new user in unified auth system."""
        try:
            unified_auth = UnifiedAuthService(db)

            if oauth_token and provider_data and len(provider_data) > 0:
                provider_type = "firebase_github"
                provider_info_data = (
                    provider_data[0]
                    if isinstance(provider_data, list)
                    else provider_data
                )
                provider_uid = provider_username or uid
            else:
                provider_type = "firebase_email"
                provider_info_data = {"email": email, "uid": uid}
                provider_uid = uid

            provider_create = AuthProviderCreate(
                provider_type=provider_type,
                provider_uid=provider_uid,
                provider_data=provider_info_data,
                access_token=oauth_token if oauth_token else None,
                is_primary=True,
            )
            unified_auth.add_provider(uid, provider_create)
            logger.info(f"Added {provider_type} provider for new user {uid}")
        except Exception as e:
            logger.warning(f"Failed to add provider for new user: {str(e)}")
            logger.warning(f"Traceback: {traceback.format_exc()}")

    @staticmethod
    def _validate_signup_request(
        body: dict,
    ) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
        """Validate signup request body and return ((uid, email), None) or (None, error)."""
        uid = body.get("uid")
        if not uid:
            return None, "uid is required"

        email = body.get("email")
        if not email:
            return None, "email is required"

        return (uid, email), None

    @staticmethod
    async def _handle_existing_user_signup(
        db: Session,
        user: any,
        uid: str,
        email: str,
        oauth_token: str,
        provider_data: list,
        provider_username: str,
        user_service: UserService,
    ) -> Response:
        """Handle signup for existing user."""
        # DO NOT update email if it's different (preserve primary sign-in email)
        if email and email.lower() != user.email.lower():
            logger.info(
                f"Email mismatch: GitHub email {email} vs existing user email "
                f"{user.email}. Keeping existing email {user.email}."
            )

        # Update last login if OAuth token provided
        if oauth_token:
            message, error = user_service.update_last_login(uid, oauth_token)
            if error:
                return Response(content=message, status_code=400)

        # Add GitHub provider if it's a GitHub signup
        if oauth_token and provider_data and len(provider_data) > 0:
            AuthAPI._handle_existing_user_github(
                db, uid, oauth_token, provider_data, provider_username
            )

        # Add email/password provider if it doesn't exist
        if not oauth_token:
            AuthAPI._handle_existing_user_email(db, uid, email)

        return Response(
            content=json.dumps({"uid": uid, "exists": True}),
            status_code=200,
        )

    @staticmethod
    async def _handle_new_user_signup(
        db: Session,
        body: dict,
        uid: str,
        email: str,
        oauth_token: str,
        provider_data: list,
        provider_username: str,
        user_service: UserService,
    ) -> Response:
        """Handle signup for new user."""
        # Check if email already exists (from SSO signup) - need to link accounts
        user_by_email = user_service.get_user_by_email(email)
        if user_by_email:
            logger.warning(
                f"Email {email} already exists with UID {user_by_email.uid}, "
                f"but new signup has UID {uid}. This should have been caught earlier."
            )
            return Response(
                content=json.dumps(
                    {"error": "Email already registered. Please sign in instead."}
                ),
                status_code=400,
            )

        # Create user
        first_login = datetime.now(timezone.utc)
        provider_info = AuthAPI._create_provider_info(
            provider_data, oauth_token, uid, email
        )
        user_email = AuthAPI._determine_user_email(
            user_service, email, oauth_token, provider_data
        )

        user = CreateUser(
            uid=uid,
            email=user_email,
            display_name=body.get("displayName", user_email.split("@")[0]),
            email_verified=body.get("emailVerified", False),
            created_at=first_login,
            last_login_at=first_login,
            provider_info=provider_info,
            provider_username=provider_username,
        )
        uid, message, error = user_service.create_user(user)

        # Check if user creation failed due to duplicate email
        if error and (
            "duplicate" in message.lower() or "already exists" in message.lower()
        ):
            link_response = AuthAPI._link_after_duplicate_error(db, uid, email)
            if link_response:
                return link_response

        # Add provider in the new system
        AuthAPI._add_provider_for_new_user(
            db, uid, email, oauth_token, provider_data, provider_username
        )

        await send_slack_message(
            f"New signup: {email} ({body.get('displayName', 'N/A')})"
        )

        PostHogClient().send_event(
            uid,
            "signup_event",
            {
                "email": email,
                "display_name": body.get("displayName", ""),
                "github_username": provider_username,
            },
        )

        if error:
            return Response(content=message, status_code=400)
        return Response(
            content=json.dumps({"uid": uid, "exists": False}),
            status_code=200,
        )

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
        try:
            body = json.loads(await request.body())
            validation_result = AuthAPI._validate_signup_request(body)
            if validation_result[1]:  # Error message
                return Response(
                    content=json.dumps({"error": validation_result[1]}),
                    status_code=400,
                )
            uid, email = validation_result[0]

            # These fields are optional (only present for GitHub OAuth signup)
            oauth_token = body.get("accessToken", "")
            provider_data = body.get("providerData", [])
            provider_username = body.get("providerUsername", "")

            user_service = UserService(db)
            user = user_service.get_user_by_uid(uid)

            # Also check by email in case user exists with different UID (e.g., SSO signup)
            user_by_email = user_service.get_user_by_email(email)

            # If user exists by email but not by UID, we need to link accounts
            if user_by_email and user_by_email.uid != uid:
                return await AuthAPI._link_accounts_by_email(
                    db, user_by_email, uid, email
                )

            if user:
                return await AuthAPI._handle_existing_user_signup(
                    db,
                    user,
                    uid,
                    email,
                    oauth_token,
                    provider_data,
                    provider_username,
                    user_service,
                )
            else:
                return await AuthAPI._handle_new_user_signup(
                    db,
                    body,
                    uid,
                    email,
                    oauth_token,
                    provider_data,
                    provider_username,
                    user_service,
                )
        except KeyError as e:
            logger.error(f"Missing required field in signup request: {str(e)}")
            return Response(
                content=json.dumps({"error": f"Missing required field: {str(e)}"}),
                status_code=400,
            )
        except Exception as e:
            logger.error(f"Error in signup endpoint: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return Response(
                content=json.dumps({"error": f"Signup failed: {str(e)}"}),
                status_code=500,
            )

    # ===== Multi-Provider SSO Endpoints =====

    @auth_router.post("/sso/login")
    async def sso_login(
        self,
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
            provider = unified_auth.get_sso_provider(sso_request.sso_provider)
            if not provider:
                logger.error(f"Unsupported SSO provider: {sso_request.sso_provider}")
                return JSONResponse(
                    content={
                        "error": f"Unsupported SSO provider: {sso_request.sso_provider}"
                    },
                    status_code=400,
                )

            # Verify token and extract user info
            provider_data = sso_request.provider_data or {}
            (
                user_info_dict,
                error_response,
            ) = await AuthAPI._verify_sso_token_and_extract_info(
                unified_auth,
                sso_request.sso_provider,
                sso_request.id_token,
                sso_request.email,
                provider_data,
            )
            if error_response:
                return error_response

            verified_email = user_info_dict["email"]
            provider_uid = user_info_dict["provider_uid"]
            display_name = user_info_dict["display_name"]
            email_verified = user_info_dict["email_verified"]
            provider_data = user_info_dict["provider_data"]

            # Authenticate or create user using verified email from token
            user, response = unified_auth.authenticate_or_create(
                email=verified_email,
                provider_type=provider_type,
                provider_uid=provider_uid,
                provider_data=provider_data,  # This includes the email for later retrieval
                display_name=display_name,
                email_verified=email_verified,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Create Firebase custom token for frontend authentication
            firebase_token = AuthAPI._create_firebase_token(user)
            if firebase_token:
                response.firebase_token = firebase_token
                logger.info(
                    f"Created Firebase custom token for user {user.uid} "
                    f"(email: {user.email})"
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

            # Include all fields including None values
            response_dict = response.model_dump(exclude_none=False)
            logger.info(
                f"SSO login response: status={response.status}, "
                f"email={verified_email}, provider={sso_request.sso_provider}"
            )
            return JSONResponse(content=response_dict, status_code=200)

        except Exception as e:
            return JSONResponse(
                content={"error": f"SSO login failed: {str(e)}"},
                status_code=400,
            )

    @auth_router.post("/providers/confirm-linking")
    async def confirm_provider_linking(
        self,
        confirm_request: ConfirmLinkingRequest,
        db: Session = Depends(get_db),
    ):
        """
        Confirm linking a new provider to existing account.

        Called after user confirms they want to link the provider
        when 'needs_linking' status is returned from login.
        """
        try:
            if not confirm_request.linking_token:
                logger.error("Missing linking_token in request")
                return JSONResponse(
                    content={"error": "linking_token is required"},
                    status_code=400,
                )

            unified_auth = UnifiedAuthService(db)
            new_provider = unified_auth.confirm_provider_link(
                confirm_request.linking_token
            )

            if not new_provider:
                logger.warning("Invalid or expired linking token")
                return JSONResponse(
                    content={
                        "error": (
                            "Invalid or expired linking token. "
                            "Please try signing in again."
                        )
                    },
                    status_code=400,
                )

            logger.info(
                f"Successfully linked provider {new_provider.provider_type} "
                f"for user {new_provider.user_id}"
            )

            provider_response = AuthProviderResponse.model_validate(
                new_provider
            ).model_dump(mode="json")

            return JSONResponse(
                content={
                    "message": "Provider linked successfully",
                    "provider": provider_response,
                },
                status_code=200,
            )

        except ValueError as e:
            logger.error(f"ValueError in confirm_provider_linking: {str(e)}")
            return JSONResponse(
                content={"error": f"Invalid request: {str(e)}"},
                status_code=400,
            )
        except Exception as e:
            logger.error(
                f"Exception in confirm_provider_linking: {str(e)}", exc_info=True
            )
            return JSONResponse(
                content={"error": f"Failed to link provider: {str(e)}"},
                status_code=400,
            )

    @auth_router.delete("/providers/cancel-linking/{linking_token}")
    async def cancel_provider_linking(
        self,
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
        self,
        request: Request,
        db: Session = Depends(get_db),
        credential: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """
        Get all authentication providers for the current user.

        Requires authentication.
        """
        try:
            # Get user from auth token
            response = Response()
            user_data = await auth_handler.check_auth(request, response, credential)
            user_id = user_data.get("user_id")

            if not user_id:
                return JSONResponse(
                    content={"error": AUTH_REQUIRED_ERROR},
                    status_code=401,
                )

            unified_auth = UnifiedAuthService(db)
            providers = unified_auth.get_user_providers(user_id)

            primary_provider = next((p for p in providers if p.is_primary), None)

            response = UserAuthProvidersResponse(
                providers=[AuthProviderResponse.model_validate(p) for p in providers],
                primary_provider=AuthProviderResponse.model_validate(primary_provider)
                if primary_provider
                else None,
            )

            return JSONResponse(
                content=response.model_dump(mode="json"),
                status_code=200,
            )

        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to get providers: {str(e)}"},
                status_code=400,
            )

    @auth_router.post("/providers/set-primary")
    async def set_primary_provider(
        self,
        request: Request,
        primary_request: SetPrimaryProviderRequest,
        db: Session = Depends(get_db),
        credential: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """Set a provider as the primary login method"""
        try:
            # Get user from auth token
            response = Response()
            user_data = await auth_handler.check_auth(request, response, credential)
            user_id = user_data.get("user_id")

            if not user_id:
                return JSONResponse(
                    content={"error": AUTH_REQUIRED_ERROR},
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
        self,
        request: Request,
        unlink_request: UnlinkProviderRequest,
        db: Session = Depends(get_db),
        credential: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """Unlink a provider from account"""
        try:
            # Get user from auth token
            response = Response()
            user_data = await auth_handler.check_auth(request, response, credential)
            user_id = user_data.get("user_id")

            if not user_id:
                return JSONResponse(
                    content={"error": AUTH_REQUIRED_ERROR},
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

    @auth_router.get("/account/check-email")
    async def check_email_providers(
        self,
        email: str,
        db: Session = Depends(get_db),
    ):
        """
        Check if an email exists and what providers it has.
        Used to help users who might have signed up with SSO.
        """
        try:
            user_service = UserService(db)
            user = user_service.get_user_by_email(email)

            if not user:
                return JSONResponse(
                    content={"exists": False, "has_sso": False},
                    status_code=200,
                )

            unified_auth = UnifiedAuthService(db)
            providers = unified_auth.get_user_providers(user.uid)

            # Check if user has SSO providers
            has_sso = any(p.provider_type.startswith("sso_") for p in providers)

            return JSONResponse(
                content={
                    "exists": True,
                    "has_sso": has_sso,
                    "providers": [p.provider_type for p in providers],
                },
                status_code=200,
            )
        except Exception as e:
            logger.error(f"Error checking email providers: {str(e)}")
            return JSONResponse(
                content={"error": "Failed to check email"},
                status_code=500,
            )

    @auth_router.get("/account/me")
    async def get_my_account(
        self,
        request: Request,
        db: Session = Depends(get_db),
        credential: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """Get complete account information including all providers"""
        try:
            # Get user from auth token
            response = Response()
            user_data = await auth_handler.check_auth(request, response, credential)
            user_id = user_data.get("user_id")

            if not user_id:
                return JSONResponse(
                    content={"error": AUTH_REQUIRED_ERROR},
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

            primary_provider_obj = next((p for p in providers if p.is_primary), None)
            primary_provider = (
                primary_provider_obj.provider_type if primary_provider_obj else None
            )

            # Determine the correct email to return based on primary provider
            display_email = AuthAPI._determine_display_email(
                user, providers, primary_provider_obj
            )

            response = AccountResponse(
                user_id=user.uid,
                email=display_email,  # Use the determined display email
                display_name=user.display_name,
                organization=user.organization,
                organization_name=user.organization_name,
                email_verified=user.email_verified,
                created_at=user.created_at,
                providers=[
                    AuthProviderResponse.model_validate(p).model_dump(mode="json")
                    for p in providers
                ],
                primary_provider=primary_provider,
            )

            return JSONResponse(
                content=response.model_dump(mode="json"),
                status_code=200,
            )

        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to get account: {str(e)}"},
                status_code=400,
            )
