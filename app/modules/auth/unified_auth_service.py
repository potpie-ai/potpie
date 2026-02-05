"""
Unified Authentication Service for Multi-Provider Support

Handles authentication across multiple providers (Firebase GitHub, SSO, etc.)
while maintaining single user identity based on email.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.modules.integrations.token_encryption import encrypt_token, decrypt_token

# Constants
LINKING_TOKEN_LENGTH = 32  # Length of URL-safe token for provider linking
LINKING_TOKEN_EXPIRY_MINUTES = 15  # Expiration time for pending provider links

# Provider Type Constants
PROVIDER_TYPE_FIREBASE_GITHUB = "firebase_github"
PROVIDER_TYPE_FIREBASE_EMAIL = "firebase_email_password"
PROVIDER_TYPE_SSO_GOOGLE = "sso_google"
PROVIDER_TYPE_SSO_AZURE = "sso_azure"
PROVIDER_TYPE_SSO_OKTA = "sso_okta"
PROVIDER_TYPE_SSO_SAML = "sso_saml"


# Use timezone-aware datetime.now() instead of deprecated utcnow()
def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime"""
    return datetime.now(timezone.utc)


from app.modules.auth.sso_providers import (
    BaseSSOProvider,
)
from app.modules.auth.sso_providers.provider_registry import SSOProviderRegistry

from app.modules.auth.auth_provider_model import (
    UserAuthProvider,
    PendingProviderLink,
    OrganizationSSOConfig,
    AuthAuditLog,
)
from app.modules.users.user_model import User
from app.modules.users.user_service import UserService
from app.modules.auth.auth_schema import (
    AuthProviderCreate,
    SSOLoginResponse,
)

logger = logging.getLogger(__name__)


class UnifiedAuthService:
    """
    Service for handling multi-provider authentication and account linking.

    Key responsibilities:
    - Authenticate users across multiple providers
    - Link new providers to existing accounts
    - Prevent account duplication
    - Manage provider preferences
    """

    def __init__(self, db: Session):
        self.db = db
        self.user_service = UserService(db)

        # Get singleton SSO provider instances from registry
        # Providers are stateless and can be safely shared across requests
        self.sso_providers: Dict[str, BaseSSOProvider] = (
            SSOProviderRegistry.get_all_providers()
        )

    def get_sso_provider(self, provider_name: str) -> Optional[BaseSSOProvider]:
        """Get SSO provider by name"""
        return self.sso_providers.get(provider_name.lower())

    async def verify_sso_token(
        self, provider_name: str, id_token: str
    ) -> Optional[Any]:
        """
        Verify an SSO ID token using the appropriate provider.

        Returns SSOUserInfo if valid, None if invalid.
        """
        provider = self.get_sso_provider(provider_name)
        if not provider:
            logger.error("Unknown SSO provider: %s", provider_name)
            return None

        try:
            user_info = await provider.verify_token(id_token)
            return user_info
        except ValueError as e:
            logger.error("Token verification failed for %s: %s", provider_name, str(e))
            return None

    # ===== Provider Management =====

    def get_user_providers(self, user_id: str) -> List[UserAuthProvider]:
        """Get all auth providers for a user"""
        return (
            self.db.query(UserAuthProvider)
            .filter(UserAuthProvider.user_id == user_id)
            .order_by(
                UserAuthProvider.is_primary.desc(), UserAuthProvider.linked_at.desc()
            )
            .all()
        )

    def get_provider(
        self, user_id: str, provider_type: str
    ) -> Optional[UserAuthProvider]:
        """Get a specific provider for a user"""
        return (
            self.db.query(UserAuthProvider)
            .filter(
                and_(
                    UserAuthProvider.user_id == user_id,
                    UserAuthProvider.provider_type == provider_type,
                )
            )
            .first()
        )

    def check_github_linked(
        self, user_id: str
    ) -> Tuple[bool, Optional[UserAuthProvider]]:
        """
        Check if a user has GitHub linked.

        Flow:
        1. Find user in users table by user_id
        2. Check user_auth_providers table for that user_id where provider_type = 'firebase_github'
        3. Return (True, provider) if found, (False, None) if not

        Args:
            user_id: The user's UID (primary key in users table)

        Returns:
            Tuple of (is_linked: bool, github_provider: Optional[UserAuthProvider])
        """
        # Step 1: Verify user exists
        user = self.db.query(User).filter(User.uid == user_id).first()
        if not user:
            logger.warning(f"User {user_id} not found in users table")
            return False, None

        # Step 2: Check user_auth_providers for GitHub provider
        github_provider = (
            self.db.query(UserAuthProvider)
            .filter(
                and_(
                    UserAuthProvider.user_id == user_id,
                    UserAuthProvider.provider_type == PROVIDER_TYPE_FIREBASE_GITHUB,
                )
            )
            .first()
        )

        if github_provider:
            logger.info(
                f"GitHub provider found for user {user_id}: "
                f"provider_id={github_provider.id}, provider_uid={github_provider.provider_uid}, "
                f"linked_at={github_provider.linked_at}"
            )
            return True, github_provider
        else:
            logger.info(f"No GitHub provider found for user {user_id}")
            return False, None

    def get_decrypted_access_token(
        self, user_id: str, provider_type: str
    ) -> Optional[str]:
        """
        Get decrypted access token for a user's provider.

        Returns None if provider not found or token not available.
        Handles both encrypted and plaintext tokens (backward compatibility).
        """
        provider = self.get_provider(user_id, provider_type)
        if not provider or not provider.access_token:
            return None

        try:
            # Try to decrypt (token is encrypted)
            return decrypt_token(provider.access_token)
        except Exception:
            # Token might be plaintext (from before encryption was added)
            # Return as-is for backward compatibility
            logger.warning(
                f"Failed to decrypt token for user {user_id}, provider {provider_type}. "
                "Assuming plaintext token (backward compatibility)."
            )
            return provider.access_token

    def get_decrypted_refresh_token(
        self, user_id: str, provider_type: str
    ) -> Optional[str]:
        """
        Get decrypted refresh token for a user's provider.

        Returns None if provider not found or token not available.
        Handles both encrypted and plaintext tokens (backward compatibility).
        """
        provider = self.get_provider(user_id, provider_type)
        if not provider or not provider.refresh_token:
            return None

        try:
            # Try to decrypt (token is encrypted)
            return decrypt_token(provider.refresh_token)
        except Exception:
            # Token might be plaintext (from before encryption was added)
            # Return as-is for backward compatibility
            logger.warning(
                "Failed to decrypt refresh token for user %s, provider %s. "
                "Assuming plaintext token (backward compatibility).",
                user_id,
                provider_type,
            )
            return provider.refresh_token

    def add_provider(
        self,
        user_id: str,
        provider_create: AuthProviderCreate,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> UserAuthProvider:
        """
        Add a new authentication provider to a user's account.

        If this is the first provider, it becomes primary.
        """
        # Check if provider already exists
        existing = self.get_provider(user_id, provider_create.provider_type)
        if existing:
            logger.warning(
                "Provider %s already exists for user %s",
                provider_create.provider_type,
                user_id,
            )
            return existing

        # Check if this is the first provider
        existing_providers = self.get_user_providers(user_id)
        is_first = len(existing_providers) == 0

        # Encrypt tokens before storing
        encrypted_access_token = (
            encrypt_token(provider_create.access_token)
            if provider_create.access_token
            else None
        )
        encrypted_refresh_token = (
            encrypt_token(provider_create.refresh_token)
            if provider_create.refresh_token
            else None
        )

        # Create new provider
        new_provider = UserAuthProvider(
            user_id=user_id,
            provider_type=provider_create.provider_type,
            provider_uid=provider_create.provider_uid,
            provider_data=provider_create.provider_data,
            access_token=encrypted_access_token,
            refresh_token=encrypted_refresh_token,
            token_expires_at=provider_create.token_expires_at,
            is_primary=is_first or provider_create.is_primary,
            linked_at=utc_now(),
            last_used_at=utc_now(),
            linked_by_ip=ip_address,
            linked_by_user_agent=user_agent,
        )

        self.db.add(new_provider)

        # If setting as primary, unset other primary providers
        if new_provider.is_primary:
            self.db.query(UserAuthProvider).filter(
                and_(
                    UserAuthProvider.user_id == user_id,
                    UserAuthProvider.id != new_provider.id,
                )
            ).update({"is_primary": False})

        self.db.commit()
        self.db.refresh(new_provider)

        # Audit log
        self._log_auth_event(
            user_id=user_id,
            event_type="link_provider",
            provider_type=provider_create.provider_type,
            status="success",
            ip_address=ip_address,
            user_agent=user_agent,
        )

        logger.info(
            "Added provider %s for user %s",
            provider_create.provider_type,
            user_id,
        )
        return new_provider

    def set_primary_provider(self, user_id: str, provider_type: str) -> bool:
        """Set a provider as the primary login method"""
        provider = self.get_provider(user_id, provider_type)
        if not provider:
            logger.warning("Provider %s not found for user %s", provider_type, user_id)
            return False

        # Unset all other primary providers
        self.db.query(UserAuthProvider).filter(
            UserAuthProvider.user_id == user_id
        ).update({"is_primary": False})

        # Set this as primary
        provider.is_primary = True
        self.db.commit()

        logger.info("Set primary provider to %s for user %s", provider_type, user_id)
        return True

    def unlink_provider(self, user_id: str, provider_type: str) -> bool:
        """
        Unlink a provider from a user's account.

        Cannot unlink if it's the only provider (would lock user out).
        If unlinking primary provider, automatically set another as primary.
        """
        provider = self.get_provider(user_id, provider_type)
        if not provider:
            logger.warning("Provider %s not found for user %s", provider_type, user_id)
            return False

        # Check if this is the only provider
        all_providers = self.get_user_providers(user_id)
        if len(all_providers) <= 1:
            logger.error(
                "Cannot unlink last provider %s for user %s",
                provider_type,
                user_id,
            )
            raise ValueError("Cannot unlink the only authentication provider")

        was_primary = provider.is_primary

        # Delete the provider
        self.db.delete(provider)
        self.db.commit()

        # If it was primary, set another as primary
        if was_primary:
            remaining = self.get_user_providers(user_id)
            if remaining:
                remaining[0].is_primary = True
                self.db.commit()

        # Audit log
        self._log_auth_event(
            user_id=user_id,
            event_type="unlink_provider",
            provider_type=provider_type,
            status="success",
        )

        logger.info("Unlinked provider %s for user %s", provider_type, user_id)
        return True

    def update_last_used(self, user_id: str, provider_type: str):
        """Update last_used_at for a provider"""
        provider = self.get_provider(user_id, provider_type)
        if provider:
            provider.last_used_at = utc_now()
            self.db.commit()

    # ===== Authentication Flow =====

    async def authenticate_or_create(
        self,
        email: str,
        provider_type: str,
        provider_uid: str,
        provider_data: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
        display_name: Optional[str] = None,
        email_verified: bool = False,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[User, SSOLoginResponse]:
        """
        Main authentication flow for any provider.

        Firebase is the primary source of truth. Local DB is synced from Firebase.

        Three scenarios:
        1. User exists with this provider → Login
        2. User exists but not this provider → Create pending link
        3. User doesn't exist → Create new user

        Returns: (User, SSOLoginResponse)
        """
        email = email.lower().strip()

        # Check if user exists by email in local DB
        existing_user = await self.user_service.get_user_by_email(email)

        if existing_user:
            # Check what providers the user has
            user_providers = self.get_user_providers(existing_user.uid)

            # All Firebase-based providers (including SSO via Firebase)
            # Now that we use Firebase ID tokens for SSO, all providers use Firebase UIDs
            firebase_based_providers = [
                p
                for p in user_providers
                if p.provider_type
                in [PROVIDER_TYPE_FIREBASE_GITHUB, PROVIDER_TYPE_FIREBASE_EMAIL]
                or p.provider_type.startswith(
                    "sso_"
                )  # sso_google, sso_azure, etc. - now use Firebase UIDs
            ]

            # Check if UID looks like a Firebase UID (28 characters, alphanumeric)
            uid_looks_like_firebase = (
                len(existing_user.uid) == 28
                and existing_user.uid.replace("_", "").replace("-", "").isalnum()
            )

            # Verify Firebase if user has Firebase-based providers OR if UID looks like Firebase UID
            if firebase_based_providers or uid_looks_like_firebase:
                from firebase_admin import auth
                from firebase_admin.exceptions import NotFoundError
                import firebase_admin

                firebase_user_exists = False
                try:
                    # Check if Firebase is initialized
                    try:
                        # Try to get the default app - this will raise ValueError if not initialized
                        firebase_admin.get_app()
                        firebase_initialized = True
                    except (ValueError, Exception) as init_error:
                        # Firebase not initialized (e.g., development mode)
                        firebase_initialized = False
                        logger.warning(
                            f"Firebase not initialized. Skipping Firebase verification for user {existing_user.uid}. "
                            f"This is normal in development mode. Error: {str(init_error)}"
                        )
                        # Assume user exists to avoid breaking functionality in dev mode
                        firebase_user_exists = True
                        firebase_initialized = (
                            False  # Ensure we don't try to use Firebase
                        )

                    if firebase_initialized:
                        try:
                            # Verify the Firebase user exists
                            auth.get_user(existing_user.uid)
                            firebase_user_exists = True
                            logger.debug(
                                f"Verified Firebase user {existing_user.uid} exists for email {email}"
                            )
                        except NotFoundError:
                            # User deleted from Firebase but exists in local DB - orphaned record
                            logger.warning(
                                f"User {existing_user.uid} with email {email} exists in local DB "
                                "but not in Firebase. Treating as orphaned record."
                            )
                            firebase_user_exists = False
                        except Exception as e:
                            # Handle case where Firebase is not properly initialized
                            if "does not exist" in str(e) or "initialize_app" in str(e):
                                logger.warning(
                                    f"Firebase not initialized when verifying user {existing_user.uid}. "
                                    "Assuming user exists (development mode)."
                                )
                                firebase_user_exists = True
                            else:
                                logger.error(
                                    f"Error verifying Firebase user {existing_user.uid}: {str(e)}"
                                )
                                # On error, assume Firebase user exists to avoid breaking existing users
                                firebase_user_exists = True
                    # If firebase_initialized is False, firebase_user_exists is already set to True above
                except Exception as e:
                    # Catch any unexpected errors
                    if "does not exist" in str(e) or "initialize_app" in str(e):
                        logger.warning(
                            f"Firebase not initialized. Assuming user {existing_user.uid} exists (development mode)."
                        )
                        firebase_user_exists = True
                    else:
                        logger.error(
                            f"Unexpected error verifying Firebase user {existing_user.uid}: {str(e)}"
                        )
                        # On error, assume Firebase user exists to avoid breaking existing users
                        firebase_user_exists = True

                if not firebase_user_exists:
                    # Orphaned record - delete from local DB and treat as new user
                    # Store UID before any operations to avoid accessing after rollback
                    user_uid = existing_user.uid
                    logger.info(
                        f"Deleting orphaned user record {user_uid} "
                        f"for email {email} (not in Firebase)"
                    )
                    try:
                        # Use helper function to delete orphaned user and all related records
                        self._delete_orphaned_user(user_uid, existing_user)

                        # Re-query to ensure user is actually deleted and not in session
                        # This prevents issues with stale SQLAlchemy objects
                        self.db.expire_all()
                        existing_user = await self.user_service.get_user_by_email(email)

                        # If user still exists (shouldn't happen), log warning
                        if existing_user:
                            logger.warning(
                                f"User {email} still exists after deletion attempt. "
                                "This may indicate a transaction issue."
                            )
                            # Force expunge and set to None
                            self.db.expunge(existing_user)
                            existing_user = None
                        else:
                            logger.info(f"Confirmed user {email} deleted successfully")
                    except Exception as e:
                        logger.error(
                            f"Failed to delete orphaned user {user_uid}: {str(e)}",
                            exc_info=True,
                        )
                        self.db.rollback()
                        # Expunge the user object from session to avoid stale state
                        self.db.expunge(existing_user)
                        # Reset to None to trigger new user creation
                        existing_user = None

        if existing_user:
            # Check if this provider is already linked
            existing_provider = self.get_provider(existing_user.uid, provider_type)

            if existing_provider:
                # Scenario 1: User exists with this provider → Login
                # BUT: Check GitHub linking FIRST before completing login

                # Refresh the user object to ensure we have the latest data
                self.db.refresh(existing_user)

                # Query for GitHub provider - check all providers for this user for debugging
                all_providers = self.get_user_providers(existing_user.uid)
                provider_types = [p.provider_type for p in all_providers]
                logger.info(
                    f"User {existing_user.uid} ({email}) has providers: {provider_types}"
                )

                # CRITICAL: Check GitHub linking using systematic approach
                # Flow: 1. Find user in users table by user_id
                #       2. Check user_auth_providers for that user_id where provider_type = 'firebase_github'
                #       3. If found → GitHub linked, if not → GitHub not linked
                has_github, github_provider = self.check_github_linked(
                    existing_user.uid
                )

                if not has_github:
                    logger.warning(
                        f"No GitHub provider found for user {existing_user.uid} ({email}). "
                        f"Available providers: {provider_types}"
                    )

                if not has_github:
                    # GitHub not linked - don't complete login, redirect to onboarding
                    logger.info(
                        f"User {existing_user.uid} ({email}) authenticated but GitHub not linked. "
                        "Requiring GitHub linking before login completion."
                    )

                    # Update last login time but don't commit yet (will commit after GitHub linking)
                    existing_user.last_login_at = utc_now()

                    # Audit log
                    self._log_auth_event(
                        user_id=existing_user.uid,
                        event_type="login_blocked_github",
                        provider_type=provider_type,
                        status="pending",
                        ip_address=ip_address,
                        user_agent=user_agent,
                    )

                    return (
                        existing_user,
                        SSOLoginResponse(
                            status="success",  # Keep as success for compatibility
                            user_id=existing_user.uid,
                            email=email,
                            display_name=existing_user.display_name,
                            message="Login successful, but GitHub account linking required",
                            needs_github_linking=True,  # Frontend will redirect to onboarding
                        ),
                    )

                # GitHub is linked - proceed with normal login
                self.update_last_used(existing_user.uid, provider_type)

                # Set this provider as primary since user is signing in with it
                # This ensures the correct email is shown in the sidebar
                if not existing_provider.is_primary:
                    logger.info(
                        "Setting %s as primary provider for user %s (user signed in with this provider)",
                        provider_type,
                        existing_user.uid,
                    )
                    self.set_primary_provider(existing_user.uid, provider_type)

                # Update last login
                existing_user.last_login_at = utc_now()
                self.db.commit()

                # Audit log
                self._log_auth_event(
                    user_id=existing_user.uid,
                    event_type="login",
                    provider_type=provider_type,
                    status="success",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )

                return existing_user, SSOLoginResponse(
                    status="success",
                    user_id=existing_user.uid,
                    email=email,
                    display_name=existing_user.display_name,
                    message="Login successful",
                    needs_github_linking=False,  # Explicitly set to False
                )
            else:
                # Scenario 2: User exists but not this provider → Create pending link
                existing_providers = [
                    p.provider_type for p in self.get_user_providers(existing_user.uid)
                ]

                linking_token = self._create_pending_link(
                    user_id=existing_user.uid,
                    provider_type=provider_type,
                    provider_uid=provider_uid,
                    provider_data=provider_data or {},
                    ip_address=ip_address,
                    user_agent=user_agent,
                )

                # Audit log
                self._log_auth_event(
                    user_id=existing_user.uid,
                    event_type="needs_linking",
                    provider_type=provider_type,
                    status="pending",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )

                providers_str = ", ".join(existing_providers)
                return existing_user, SSOLoginResponse(
                    status="needs_linking",
                    user_id=existing_user.uid,
                    email=email,
                    display_name=existing_user.display_name,
                    message=f"Account exists with {providers_str}. Link this provider?",
                    linking_token=linking_token,
                    existing_providers=existing_providers,
                )
        else:
            # Scenario 3: User doesn't exist → Create new user
            # Double-check that user doesn't exist (in case of race condition or stale session)
            final_check = await self.user_service.get_user_by_email(email)
            if final_check:
                logger.warning(
                    f"User with email {email} found during final check before creation. "
                    f"UID: {final_check.uid}. This may indicate a race condition."
                )
                # User exists - treat as existing user scenario
                existing_user = final_check
                # Check if this provider is already linked
                existing_provider = self.get_provider(existing_user.uid, provider_type)

                if existing_provider:
                    # User exists with this provider → Login
                    # BUT: Check GitHub linking FIRST before completing login
                    # Use systematic check_github_linked method
                    has_github, github_provider = self.check_github_linked(
                        existing_user.uid
                    )

                    if not has_github:
                        # GitHub not linked - require linking before login completion
                        logger.info(
                            f"User {existing_user.uid} ({email}) authenticated but GitHub not linked. "
                            "Requiring GitHub linking before login completion (race condition path)."
                        )
                        existing_user.last_login_at = utc_now()

                        return (
                            existing_user,
                            SSOLoginResponse(
                                status="success",  # Keep as success for compatibility
                                user_id=existing_user.uid,
                                email=email,
                                display_name=existing_user.display_name,
                                message="Login successful, but GitHub account linking required",
                                needs_github_linking=True,  # Frontend will redirect to onboarding
                            ),
                        )

                    # GitHub is linked - proceed with normal login
                    self.update_last_used(existing_user.uid, provider_type)
                    existing_user.last_login_at = utc_now()
                    self.db.commit()

                    return existing_user, SSOLoginResponse(
                        status="success",
                        user_id=existing_user.uid,
                        email=email,
                        display_name=existing_user.display_name,
                        message="Login successful",
                        needs_github_linking=False,  # Explicitly set to False
                    )
                else:
                    # User exists but not this provider → Create pending link
                    existing_providers = [
                        p.provider_type
                        for p in self.get_user_providers(existing_user.uid)
                    ]
                    linking_token = self._create_pending_link(
                        user_id=existing_user.uid,
                        provider_type=provider_type,
                        provider_uid=provider_uid,
                        provider_data=provider_data or {},
                        ip_address=ip_address,
                        user_agent=user_agent,
                    )

                    return existing_user, SSOLoginResponse(
                        status="needs_linking",
                        user_id=existing_user.uid,
                        email=email,
                        display_name=existing_user.display_name,
                        message="Account exists. Link this provider?",
                        linking_token=linking_token,
                        existing_providers=existing_providers,
                    )

            # User truly doesn't exist - create new user
            new_user = self._create_user_with_provider(
                email=email,
                provider_type=provider_type,
                provider_uid=provider_uid,
                provider_data=provider_data,
                access_token=access_token,
                display_name=display_name,
                email_verified=email_verified,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Check GitHub linking for new users (new users won't have GitHub linked yet)
            github_provider = (
                self.db.query(UserAuthProvider)
                .filter(
                    UserAuthProvider.user_id == new_user.uid,
                    UserAuthProvider.provider_type == PROVIDER_TYPE_FIREBASE_GITHUB,
                )
                .first()
            )

            # Only check for GitHub provider in database - don't rely on provider_username
            # as it may exist for other providers (e.g., Google SSO)
            has_github = github_provider is not None

            # Audit log
            self._log_auth_event(
                user_id=new_user.uid,
                event_type="signup",
                provider_type=provider_type,
                status="success",
                ip_address=ip_address,
                user_agent=user_agent,
            )

            return new_user, SSOLoginResponse(
                status="new_user",
                user_id=new_user.uid,
                email=email,
                display_name=new_user.display_name,
                message="Account created successfully",
                needs_github_linking=not has_github,  # Set to True if GitHub not linked
            )

    # ===== Pending Provider Links =====

    def _create_pending_link(
        self,
        user_id: str,
        provider_type: str,
        provider_uid: str,
        provider_data: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Create a pending provider link (expires in 15 minutes)"""
        from sqlalchemy.exc import IntegrityError, InternalError

        try:
            token = secrets.token_urlsafe(LINKING_TOKEN_LENGTH)
            expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=LINKING_TOKEN_EXPIRY_MINUTES
            )

            pending_link = PendingProviderLink(
                user_id=user_id,
                provider_type=provider_type,
                provider_uid=provider_uid,
                provider_data=provider_data,
                token=token,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            self.db.add(pending_link)
            self.db.commit()

            logger.info(
                "Created pending link for user %s, provider %s",
                user_id,
                provider_type,
            )
            return token
        except (IntegrityError, InternalError) as e:
            self.db.rollback()
            logger.error(
                f"Database error creating pending link for user_id={user_id}, provider_type={provider_type}, provider_uid={provider_uid}: {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(
                f"Unexpected error creating pending link for user_id={user_id}, provider_type={provider_type}, provider_uid={provider_uid}: {e}",
                exc_info=True,
            )
            raise

    def confirm_provider_link(self, linking_token: str) -> Optional[UserAuthProvider]:
        """
        Confirm and complete a pending provider link.

        Returns the newly linked provider or None if token invalid/expired.
        """
        pending = (
            self.db.query(PendingProviderLink)
            .filter(PendingProviderLink.token == linking_token)
            .first()
        )

        if not pending:
            logger.warning("Invalid linking token: %s", linking_token)
            return None

        # Check expiration - ensure both datetimes are timezone-aware
        now = datetime.now(timezone.utc)
        expires_at = pending.expires_at
        # If expires_at is naive, make it timezone-aware (assume UTC)
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        # If expires_at is timezone-aware, ensure now is also timezone-aware
        elif expires_at is not None and expires_at.tzinfo is not None:
            # Both are timezone-aware, comparison should work
            pass
        logger.info(
            f"Current time (UTC): {now}, Expires at: {expires_at}, "
            f"expires_at.tzinfo: {expires_at.tzinfo if expires_at else None}"
        )
        if expires_at and expires_at < now:
            logger.warning(
                "Expired linking token: %s (expired at %s, now is %s)",
                linking_token,
                expires_at,
                now,
            )
            self.db.delete(pending)
            self.db.commit()
            return None

        logger.info("Token is valid, checking if provider already exists...")
        # Check if provider already exists (might have been added via signup endpoint)
        existing_provider = self.get_provider(pending.user_id, pending.provider_type)
        if existing_provider:
            logger.info(
                "Provider %s already exists for user %s. Provider ID: %s. Deleting pending link.",
                pending.provider_type,
                pending.user_id,
                existing_provider.id,
            )
            self.db.delete(pending)
            self.db.commit()
            return existing_provider

        logger.info("Provider doesn't exist, creating new provider...")
        try:
            # Check if user already has providers (to determine if this should be primary)
            existing_providers = self.get_user_providers(pending.user_id)
            has_existing_providers = len(existing_providers) > 0

            # When linking a provider, don't make it primary if user already has providers
            # This preserves the original sign-in method's email in the sidebar
            is_primary = (
                not has_existing_providers
            )  # Only primary if it's the first provider

            # Create the provider
            provider_create = AuthProviderCreate(
                provider_type=pending.provider_type,
                provider_uid=pending.provider_uid,
                provider_data=pending.provider_data,
                is_primary=is_primary,  # Explicitly set based on whether user has existing providers
            )
            new_provider = self.add_provider(
                user_id=pending.user_id,
                provider_create=provider_create,
                ip_address=pending.ip_address,
                user_agent=pending.user_agent,
            )

            # Delete the pending link
            self.db.delete(pending)
            self.db.commit()

            logger.info(
                "Confirmed provider link for user %s, provider %s",
                pending.user_id,
                pending.provider_type,
            )
            return new_provider
        except Exception as e:
            logger.error("Error confirming provider link: %s", str(e), exc_info=True)
            self.db.rollback()
            raise

    def cancel_pending_link(self, linking_token: str) -> bool:
        """Cancel a pending provider link"""
        pending = (
            self.db.query(PendingProviderLink)
            .filter(PendingProviderLink.token == linking_token)
            .first()
        )

        if pending:
            self.db.delete(pending)
            self.db.commit()
            logger.info("Cancelled pending link: %s", linking_token)
            return True

        return False

    # ===== Helper Methods =====

    def _delete_orphaned_user(self, user_uid: str, user_obj: User) -> None:
        """
        Delete an orphaned user and all related records.

        Handles deletion of all child tables that don't have CASCADE delete,
        in the correct order to avoid foreign key constraint violations.

        Args:
            user_uid: The user's UID to delete
            user_obj: The User SQLAlchemy object (will be expunged)
        """
        from sqlalchemy import text

        try:
            # Step 1: Delete user-owned prompts (created_by FK, no CASCADE)
            from app.modules.intelligence.prompts.prompt_model import Prompt

            prompts_count = (
                self.db.query(Prompt).filter(Prompt.created_by == user_uid).count()
            )
            if prompts_count > 0:
                logger.info(
                    f"Deleting {prompts_count} prompts created by orphaned user {user_uid}"
                )
                self.db.execute(
                    text("DELETE FROM prompts WHERE created_by = :user_id"),
                    {"user_id": user_uid},
                )

            # Step 2: Delete user preferences (user_id FK, no CASCADE)
            from app.modules.users.user_preferences_model import UserPreferences

            preferences_count = (
                self.db.query(UserPreferences)
                .filter(UserPreferences.user_id == user_uid)
                .count()
            )
            if preferences_count > 0:
                logger.info(f"Deleting user preferences for orphaned user {user_uid}")
                self.db.execute(
                    text("DELETE FROM user_preferences WHERE user_id = :user_id"),
                    {"user_id": user_uid},
                )

            # Step 3: Delete OrganizationSSOConfig entries (configured_by FK, no CASCADE)
            from app.modules.auth.auth_provider_model import OrganizationSSOConfig

            org_configs_count = (
                self.db.query(OrganizationSSOConfig)
                .filter(OrganizationSSOConfig.configured_by == user_uid)
                .count()
            )
            if org_configs_count > 0:
                logger.info(
                    f"Deleting {org_configs_count} organization SSO configs for orphaned user {user_uid}"
                )
                self.db.execute(
                    text(
                        "DELETE FROM organization_sso_config WHERE configured_by = :user_id"
                    ),
                    {"user_id": user_uid},
                )

            # Step 4: Delete search_indices (they reference projects, must be deleted first)
            from app.modules.search.search_models import SearchIndex
            from app.modules.projects.projects_model import Project

            search_indices_count = (
                self.db.query(SearchIndex)
                .join(Project, SearchIndex.project_id == Project.id)
                .filter(Project.user_id == user_uid)
                .count()
            )
            if search_indices_count > 0:
                logger.info(
                    f"Deleting {search_indices_count} search indices for orphaned user {user_uid}"
                )
                self.db.execute(
                    text("""
                        DELETE FROM search_indices
                        WHERE project_id IN (
                            SELECT id FROM projects WHERE user_id = :user_id
                        )
                    """),
                    {"user_id": user_uid},
                )

            # Step 5: Delete projects (user_id FK with CASCADE, but we delete explicitly for clarity)
            projects_count = (
                self.db.query(Project).filter(Project.user_id == user_uid).count()
            )
            if projects_count > 0:
                logger.info(
                    f"Deleting {projects_count} projects for orphaned user {user_uid}"
                )
                self.db.execute(
                    text("DELETE FROM projects WHERE user_id = :user_id"),
                    {"user_id": user_uid},
                )

            # Step 6: Delete custom agents (user_id FK, no CASCADE)
            from app.modules.intelligence.agents.custom_agents.custom_agent_model import (
                CustomAgent,
            )

            custom_agents_count = (
                self.db.query(CustomAgent)
                .filter(CustomAgent.user_id == user_uid)
                .count()
            )
            if custom_agents_count > 0:
                logger.info(
                    f"Deleting {custom_agents_count} custom agents for orphaned user {user_uid}"
                )
                self.db.execute(
                    text("DELETE FROM custom_agents WHERE user_id = :user_id"),
                    {"user_id": user_uid},
                )

            # Step 7: Delete conversations (user_id FK with CASCADE, but we delete explicitly)
            from app.modules.conversations.conversation.conversation_model import (
                Conversation,
            )

            conversations_count = (
                self.db.query(Conversation)
                .filter(Conversation.user_id == user_uid)
                .count()
            )
            if conversations_count > 0:
                logger.info(
                    f"Deleting {conversations_count} conversations for orphaned user {user_uid}"
                )
                self.db.execute(
                    text("DELETE FROM conversations WHERE user_id = :user_id"),
                    {"user_id": user_uid},
                )

            # Step 8: Flush to ensure all deletes are executed before deleting user
            self.db.flush()

            # Step 9: Expunge the user object to avoid SQLAlchemy trying to update related objects
            self.db.expunge(user_obj)

            # Step 10: Delete user using raw SQL to avoid relationship tracking issues
            self.db.execute(
                text("DELETE FROM users WHERE uid = :user_id"),
                {"user_id": user_uid},
            )
            self.db.commit()
            logger.info(
                f"Successfully deleted orphaned user {user_uid} and all related records"
            )

        except Exception as e:
            logger.error(
                f"Error in _delete_orphaned_user for {user_uid}: {str(e)}",
                exc_info=True,
            )
            self.db.rollback()
            # Re-expunge user object after rollback
            try:
                self.db.expunge(user_obj)
            except Exception:
                pass  # Object may already be expunged
            raise  # Re-raise to be handled by caller

    def _create_user_with_provider(
        self,
        email: str,
        provider_type: str,
        provider_uid: str,
        provider_data: Optional[Dict[str, Any]],
        access_token: Optional[str],
        display_name: Optional[str],
        email_verified: bool,
        ip_address: Optional[str],
        user_agent: Optional[str],
    ) -> User:
        """Create a new user with their first auth provider"""
        from sqlalchemy.exc import IntegrityError, InternalError
        from app.modules.utils.email_helper import extract_organization_from_email

        try:
            organization = extract_organization_from_email(email)

            # Create user
            new_user = User(
                uid=provider_uid,  # Use provider UID as user ID initially
                email=email,
                display_name=display_name or email.split("@")[0],
                email_verified=email_verified,
                organization=organization,
                created_at=utc_now(),
                last_login_at=utc_now(),
            )

            self.db.add(new_user)
            self.db.flush()  # Get the user ID

            # Encrypt token before storing
            encrypted_access_token = (
                encrypt_token(access_token) if access_token else None
            )

            # Create provider
            provider = UserAuthProvider(
                user_id=new_user.uid,
                provider_type=provider_type,
                provider_uid=provider_uid,
                provider_data=provider_data,
                access_token=encrypted_access_token,
                is_primary=True,  # First provider is always primary
                linked_at=utc_now(),
                last_used_at=utc_now(),
                linked_by_ip=ip_address,
                linked_by_user_agent=user_agent,
            )

            self.db.add(provider)
            self.db.commit()
            self.db.refresh(new_user)

            logger.info(
                "Created new user %s with provider %s", new_user.uid, provider_type
            )
            return new_user
        except (IntegrityError, InternalError) as e:
            # Rollback the transaction on database errors
            self.db.rollback()
            logger.error(
                f"Database error creating user with provider {provider_type}, email={email}, provider_uid={provider_uid}: {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            # Rollback on any other error
            self.db.rollback()
            logger.error(
                f"Unexpected error creating user with provider {provider_type}, email={email}, provider_uid={provider_uid}: {e}",
                exc_info=True,
            )
            raise

    def _log_auth_event(
        self,
        user_id: Optional[str],
        event_type: str,
        provider_type: Optional[str],
        status: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Log an authentication event to audit log"""
        log_entry = AuthAuditLog(
            user_id=user_id,
            event_type=event_type,
            provider_type=provider_type,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message,
        )

        self.db.add(log_entry)
        self.db.commit()

    # ===== Organization SSO Config =====

    def get_org_sso_config(self, domain: str) -> Optional[OrganizationSSOConfig]:
        """Get SSO configuration for a domain"""
        return (
            self.db.query(OrganizationSSOConfig)
            .filter(
                and_(
                    OrganizationSSOConfig.domain == domain,
                    OrganizationSSOConfig.is_active,
                )
            )
            .first()
        )

    def should_enforce_sso(self, email: str) -> Tuple[bool, Optional[str]]:
        """
        Check if SSO should be enforced for this email.

        Returns: (enforce_sso, sso_provider)
        """
        email_domain = email.split("@")[1] if "@" in email else None
        if not email_domain:
            return False, None

        config = self.get_org_sso_config(email_domain)
        if config and config.enforce_sso:
            return True, config.sso_provider

        return False, None
