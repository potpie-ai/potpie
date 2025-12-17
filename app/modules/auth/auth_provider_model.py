"""
Authentication Provider Models for SSO Support

This module contains models for managing multiple authentication providers per user,
enabling SSO alongside existing GitHub OAuth while preventing account duplication.
"""

import uuid
from sqlalchemy import (
    Column,
    String,
    Boolean,
    TIMESTAMP,
    ForeignKey,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class UserAuthProvider(Base):
    """
    Stores multiple authentication providers for a single user.
    
    A user can have multiple auth providers (GitHub, Google SSO, Azure AD, etc.)
    but maintains a single account identity based on email.
    
    Examples:
        - User signs up with GitHub → Creates firebase_github provider
        - Same user later adds SSO → Creates sso_google provider
        - Both providers linked to same user account
    """

    __tablename__ = "user_auth_providers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="CASCADE"),
        nullable=False,
    )

    # Provider information
    provider_type = Column(
        String(50), nullable=False
    )  # 'firebase_github', 'sso_google', 'sso_azure', 'sso_okta', 'sso_saml'
    provider_uid = Column(
        String(255), nullable=False
    )  # Provider's unique ID for this user
    provider_data = Column(JSONB)  # Full provider response/metadata

    # OAuth tokens (for API access, e.g., GitHub API)
    access_token = Column(Text)  # OAuth access token (should be encrypted in production)
    refresh_token = Column(Text)  # OAuth refresh token (should be encrypted)
    token_expires_at = Column(TIMESTAMP(timezone=True))

    # Metadata
    is_primary = Column(
        Boolean, default=False
    )  # Primary login method (user's preference)
    linked_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    last_used_at = Column(TIMESTAMP(timezone=True))

    # Audit information
    linked_by_ip = Column(String(45))  # IP address when linked
    linked_by_user_agent = Column(Text)  # User agent when linked

    # Relationships
    user = relationship("User", back_populates="auth_providers")

    __table_args__ = (
        UniqueConstraint("user_id", "provider_type", name="unique_user_provider"),
        UniqueConstraint("provider_type", "provider_uid", name="unique_provider_uid"),
    )

    def __repr__(self):
        return f"<UserAuthProvider(id={self.id}, user_id={self.user_id}, provider={self.provider_type}, is_primary={self.is_primary})>"


class PendingProviderLink(Base):
    """
    Temporary storage for provider linking confirmation.
    
    When a user attempts to login with a new provider but already has an account
    with the same email, we create a pending link and ask for user confirmation.
    
    These records expire after 15 minutes for security.
    """

    __tablename__ = "pending_provider_links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="CASCADE"),
        nullable=False,
    )

    # Provider information to be linked
    provider_type = Column(String(50), nullable=False)
    provider_uid = Column(String(255), nullable=False)
    provider_data = Column(JSONB, nullable=False)

    # Confirmation token
    token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)

    # Request context for security
    ip_address = Column(String(45))
    user_agent = Column(Text)

    def __repr__(self):
        return f"<PendingProviderLink(id={self.id}, user_id={self.user_id}, provider={self.provider_type}, expires={self.expires_at})>"


class OrganizationSSOConfig(Base):
    """
    SSO configuration for organizational domains.
    
    Maps email domains to SSO providers and stores configuration.
    Enables features like:
    - Auto-detecting which SSO provider to use based on email domain
    - Enforcing SSO for specific domains (if required by organization)
    - Storing provider-specific settings
    """

    __tablename__ = "organization_sso_config"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain = Column(String(255), unique=True, nullable=False)  # e.g., "acme.com"
    organization_name = Column(String(255))  # e.g., "Acme Corporation"

    # SSO Configuration
    sso_provider = Column(
        String(50), nullable=False
    )  # 'google', 'azure', 'okta', 'saml'
    sso_config = Column(JSONB, nullable=False)  # Provider-specific configuration

    # Policies
    enforce_sso = Column(
        Boolean, default=False
    )  # If true, force SSO for this domain
    allow_other_providers = Column(
        Boolean, default=True
    )  # Allow GitHub, etc. alongside SSO

    # Metadata
    configured_by = Column(String(255), ForeignKey("users.uid"))
    configured_at = Column(
        TIMESTAMP(timezone=True), default=func.now(), nullable=False
    )
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<OrganizationSSOConfig(domain={self.domain}, provider={self.sso_provider}, enforce={self.enforce_sso})>"


class AuthAuditLog(Base):
    """
    Audit log for all authentication and authorization events.
    
    Tracks:
    - Login attempts (success/failure)
    - Provider linking/unlinking
    - SSO authentications
    - Security events
    
    Critical for:
    - Security monitoring
    - Compliance (SOC 2, GDPR)
    - Incident investigation
    """

    __tablename__ = "auth_audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), ForeignKey("users.uid", ondelete="SET NULL"))

    # Event details
    event_type = Column(
        String(50), nullable=False
    )  # 'login', 'link_provider', 'unlink_provider', 'failed_login', 'sso_auth'
    provider_type = Column(String(50))  # Which provider was used
    status = Column(String(20), nullable=False)  # 'success', 'failure', 'pending'

    # Context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    error_message = Column(Text)  # If status = 'failure'

    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    extra_data = Column(JSONB)  # Additional context if needed

    def __repr__(self):
        return f"<AuthAuditLog(event={self.event_type}, user_id={self.user_id}, status={self.status}, created={self.created_at})>"

