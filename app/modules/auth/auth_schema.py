from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID


# ===== Legacy Auth Schemas =====

class LoginRequest(BaseModel):
    email: str
    password: str


# ===== Multi-Provider Auth Schemas =====

class ProviderInfo(BaseModel):
    """Information about an authentication provider"""
    provider_id: str
    provider_name: str
    provider_email: Optional[str] = None
    
    
class AuthProviderCreate(BaseModel):
    """Create a new auth provider for a user"""
    provider_type: str = Field(..., description="e.g., 'firebase_github', 'sso_google', 'sso_azure'")
    provider_uid: str = Field(..., description="Provider's unique ID for this user")
    provider_data: Optional[Dict[str, Any]] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    is_primary: bool = False


class AuthProviderResponse(BaseModel):
    """Response model for auth provider"""
    id: UUID
    user_id: str
    provider_type: str
    provider_uid: str
    provider_data: Optional[Dict[str, Any]] = None
    is_primary: bool
    linked_at: datetime
    last_used_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserAuthProvidersResponse(BaseModel):
    """List of all providers for a user"""
    providers: List[AuthProviderResponse]
    primary_provider: Optional[AuthProviderResponse] = None


# ===== SSO Login Flow =====

class SSOLoginRequest(BaseModel):
    """Request to login via SSO provider"""
    email: str = Field(..., description="User's work email")
    sso_provider: str = Field(..., description="'google', 'azure', 'okta', 'saml'")
    id_token: str = Field(..., description="ID token from SSO provider")
    provider_data: Optional[Dict[str, Any]] = None


class SSOLoginResponse(BaseModel):
    """Response from SSO login"""
    status: str = Field(..., description="'success', 'needs_linking', 'new_user'")
    user_id: Optional[str] = None
    email: str
    display_name: Optional[str] = None
    access_token: Optional[str] = None
    message: str
    linking_token: Optional[str] = None  # If needs_linking
    existing_providers: Optional[List[str]] = None  # If needs_linking


# ===== Provider Linking Flow =====

class LinkProviderRequest(BaseModel):
    """Request to link a new provider to existing account"""
    existing_user_token: str = Field(..., description="Current user's auth token")
    new_provider_type: str = Field(..., description="Provider type to link")
    new_provider_token: str = Field(..., description="ID token from new provider")
    provider_data: Optional[Dict[str, Any]] = None
    set_as_primary: bool = False


class ConfirmLinkingRequest(BaseModel):
    """Confirm account linking"""
    linking_token: str = Field(..., description="Token from pending_provider_links")
    
    
class UnlinkProviderRequest(BaseModel):
    """Unlink a provider from account"""
    provider_type: str = Field(..., description="Provider to unlink")


# ===== Organization SSO Config =====

class OrganizationSSOConfigCreate(BaseModel):
    """Create SSO config for an organization"""
    domain: str = Field(..., description="Email domain, e.g., 'acme.com'")
    organization_name: Optional[str] = None
    sso_provider: str = Field(..., description="'google', 'azure', 'okta', 'saml'")
    sso_config: Dict[str, Any] = Field(..., description="Provider-specific configuration")
    enforce_sso: bool = False
    allow_other_providers: bool = True


class OrganizationSSOConfigResponse(BaseModel):
    """Response model for organization SSO config"""
    id: UUID
    domain: str
    organization_name: Optional[str]
    sso_provider: str
    enforce_sso: bool
    allow_other_providers: bool
    is_active: bool
    configured_at: datetime
    
    class Config:
        from_attributes = True


# ===== Account Management =====

class SetPrimaryProviderRequest(BaseModel):
    """Set a provider as primary"""
    provider_type: str
    
    
class AccountResponse(BaseModel):
    """Complete account information"""
    user_id: str
    email: str
    display_name: Optional[str]
    organization: Optional[str]
    organization_name: Optional[str]
    email_verified: bool
    created_at: datetime
    providers: List[AuthProviderResponse]
    primary_provider: Optional[str] = None


# ===== Audit Log =====

class AuthAuditLogResponse(BaseModel):
    """Auth audit log entry"""
    id: UUID
    user_id: Optional[str]
    event_type: str
    provider_type: Optional[str]
    status: str
    ip_address: Optional[str]
    created_at: datetime
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True
