"""
Unified Authentication Service for Multi-Provider Support

Handles authentication across multiple providers (Firebase GitHub, SSO, etc.)
while maintaining single user identity based on email.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.modules.auth.sso_providers import (
    GoogleSSOProvider,
    AzureSSOProvider,
    BaseSSOProvider,
)

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
    AuthProviderResponse,
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
        
        # Initialize SSO providers
        self.sso_providers: Dict[str, BaseSSOProvider] = {
            "google": GoogleSSOProvider(),
            "azure": AzureSSOProvider(),
        }
    
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
            logger.error(f"Unknown SSO provider: {provider_name}")
            return None
        
        try:
            user_info = await provider.verify_token(id_token)
            return user_info
        except ValueError as e:
            logger.error(f"Token verification failed for {provider_name}: {str(e)}")
            return None
    
    # ===== Provider Management =====
    
    def get_user_providers(self, user_id: str) -> List[UserAuthProvider]:
        """Get all auth providers for a user"""
        return (
            self.db.query(UserAuthProvider)
            .filter(UserAuthProvider.user_id == user_id)
            .order_by(UserAuthProvider.is_primary.desc(), UserAuthProvider.linked_at.desc())
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
                f"Provider {provider_create.provider_type} already exists for user {user_id}"
            )
            return existing
        
        # Check if this is the first provider
        existing_providers = self.get_user_providers(user_id)
        is_first = len(existing_providers) == 0
        
        # Create new provider
        new_provider = UserAuthProvider(
            user_id=user_id,
            provider_type=provider_create.provider_type,
            provider_uid=provider_create.provider_uid,
            provider_data=provider_create.provider_data,
            access_token=provider_create.access_token,
            refresh_token=provider_create.refresh_token,
            token_expires_at=provider_create.token_expires_at,
            is_primary=is_first or provider_create.is_primary,
            linked_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
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
            f"Added provider {provider_create.provider_type} for user {user_id}"
        )
        return new_provider
    
    def set_primary_provider(self, user_id: str, provider_type: str) -> bool:
        """Set a provider as the primary login method"""
        provider = self.get_provider(user_id, provider_type)
        if not provider:
            logger.warning(
                f"Provider {provider_type} not found for user {user_id}"
            )
            return False
        
        # Unset all other primary providers
        self.db.query(UserAuthProvider).filter(
            UserAuthProvider.user_id == user_id
        ).update({"is_primary": False})
        
        # Set this as primary
        provider.is_primary = True
        self.db.commit()
        
        logger.info(f"Set primary provider to {provider_type} for user {user_id}")
        return True
    
    def unlink_provider(self, user_id: str, provider_type: str) -> bool:
        """
        Unlink a provider from a user's account.
        
        Cannot unlink if it's the only provider (would lock user out).
        If unlinking primary provider, automatically set another as primary.
        """
        provider = self.get_provider(user_id, provider_type)
        if not provider:
            logger.warning(
                f"Provider {provider_type} not found for user {user_id}"
            )
            return False
        
        # Check if this is the only provider
        all_providers = self.get_user_providers(user_id)
        if len(all_providers) <= 1:
            logger.error(
                f"Cannot unlink last provider {provider_type} for user {user_id}"
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
        
        logger.info(f"Unlinked provider {provider_type} for user {user_id}")
        return True
    
    def update_last_used(self, user_id: str, provider_type: str):
        """Update last_used_at for a provider"""
        provider = self.get_provider(user_id, provider_type)
        if provider:
            provider.last_used_at = datetime.utcnow()
            self.db.commit()
    
    # ===== Authentication Flow =====
    
    def authenticate_or_create(
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
        
        Three scenarios:
        1. User exists with this provider → Login
        2. User exists but not this provider → Create pending link
        3. User doesn't exist → Create new user
        
        Returns: (User, SSOLoginResponse)
        """
        email = email.lower().strip()
        
        # Check if user exists by email
        existing_user = self.user_service.get_user_by_email(email)
        
        if existing_user:
            # Check if this provider is already linked
            existing_provider = self.get_provider(existing_user.uid, provider_type)
            
            if existing_provider:
                # Scenario 1: User exists with this provider → Login
                self.update_last_used(existing_user.uid, provider_type)
                
                # Update last login
                existing_user.last_login_at = datetime.utcnow()
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
                )
            else:
                # Scenario 2: User exists but not this provider → Create pending link
                existing_providers = [p.provider_type for p in self.get_user_providers(existing_user.uid)]
                
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
                
                return existing_user, SSOLoginResponse(
                    status="needs_linking",
                    user_id=existing_user.uid,
                    email=email,
                    display_name=existing_user.display_name,
                    message=f"Account exists with {', '.join(existing_providers)}. Link this provider?",
                    linking_token=linking_token,
                    existing_providers=existing_providers,
                )
        else:
            # Scenario 3: User doesn't exist → Create new user
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
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=15)
        
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
        
        logger.info(f"Created pending link for user {user_id}, provider {provider_type}")
        return token
    
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
            logger.warning(f"Invalid linking token: {linking_token}")
            return None
        
        # Check expiration
        if pending.expires_at < datetime.utcnow():
            logger.warning(f"Expired linking token: {linking_token}")
            self.db.delete(pending)
            self.db.commit()
            return None
        
        # Create the provider
        provider_create = AuthProviderCreate(
            provider_type=pending.provider_type,
            provider_uid=pending.provider_uid,
            provider_data=pending.provider_data,
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
            f"Confirmed provider link for user {pending.user_id}, "
            f"provider {pending.provider_type}"
        )
        return new_provider
    
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
            logger.info(f"Cancelled pending link: {linking_token}")
            return True
        
        return False
    
    # ===== Helper Methods =====
    
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
        # Extract organization from email
        email_domain = email.split("@")[1] if "@" in email else None
        organization = None
        
        if email_domain:
            personal_domains = {
                "gmail.com",
                "yahoo.com",
                "hotmail.com",
                "outlook.com",
                "icloud.com",
                "protonmail.com",
            }
            if email_domain not in personal_domains:
                organization = email_domain
        
        # Create user
        new_user = User(
            uid=provider_uid,  # Use provider UID as user ID initially
            email=email,
            display_name=display_name or email.split("@")[0],
            email_verified=email_verified,
            organization=organization,
            created_at=datetime.utcnow(),
            last_login_at=datetime.utcnow(),
        )
        
        self.db.add(new_user)
        self.db.flush()  # Get the user ID
        
        # Create provider
        provider = UserAuthProvider(
            user_id=new_user.uid,
            provider_type=provider_type,
            provider_uid=provider_uid,
            provider_data=provider_data,
            access_token=access_token,
            is_primary=True,  # First provider is always primary
            linked_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
            linked_by_ip=ip_address,
            linked_by_user_agent=user_agent,
        )
        
        self.db.add(provider)
        self.db.commit()
        self.db.refresh(new_user)
        
        logger.info(f"Created new user {new_user.uid} with provider {provider_type}")
        return new_user
    
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
                    OrganizationSSOConfig.is_active == True,
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

