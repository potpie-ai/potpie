from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class IntegrationType(str, Enum):
    """Supported integration types"""

    SENTRY = "sentry"
    GITHUB = "github"
    SLACK = "slack"
    JIRA = "jira"
    LINEAR = "linear"


class IntegrationStatus(str, Enum):
    """Integration status values"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ERROR = "error"


class AuthData(BaseModel):
    """Authentication data for integrations"""

    access_token: Optional[str] = Field(None, description="OAuth access token")
    refresh_token: Optional[str] = Field(None, description="OAuth refresh token")
    token_type: Optional[str] = Field("Bearer", description="Token type")
    expires_at: Optional[datetime] = Field(None, description="Token expiration time")
    scope: Optional[str] = Field(None, description="OAuth scope")
    code: Optional[str] = Field(None, description="OAuth authorization code")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() if v else None}}


class ScopeData(BaseModel):
    """Scope-specific data for integrations"""

    org_slug: Optional[str] = Field(None, description="Organization slug")
    installation_id: Optional[str] = Field(None, description="Installation ID")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")
    project_id: Optional[str] = Field(None, description="Project ID")


class IntegrationMetadata(BaseModel):
    """Additional metadata for integrations"""

    instance_name: str = Field(
        ..., description="User-defined name for this integration"
    )
    created_via: str = Field(
        default="oauth_callback", description="How the integration was created"
    )
    version: Optional[str] = Field(None, description="Integration version")
    description: Optional[str] = Field(None, description="Integration description")
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Integration tags"
    )


class Integration(BaseModel):
    """Core integration model"""

    integration_id: str = Field(..., description="Unique integration identifier")
    name: str = Field(..., description="Integration name")
    integration_type: IntegrationType = Field(..., description="Type of integration")
    status: IntegrationStatus = Field(
        default=IntegrationStatus.ACTIVE, description="Integration status"
    )
    active: bool = Field(default=True, description="Whether integration is active")

    # Authentication and scope data
    auth_data: AuthData = Field(..., description="Authentication data")
    scope_data: ScopeData = Field(..., description="Scope-specific data")
    metadata: IntegrationMetadata = Field(..., description="Integration metadata")

    # System fields
    unique_identifier: str = Field(
        ..., description="Unique identifier for the integration"
    )
    created_by: str = Field(..., description="User who created the integration")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    model_config = {
        "use_enum_values": True,
        "json_encoders": {datetime: lambda v: v.isoformat() if v else None},
    }


class IntegrationCreateRequest(BaseModel):
    """Request to create a new integration"""

    name: str = Field(..., description="Integration name")
    integration_type: IntegrationType = Field(..., description="Type of integration")
    auth_data: AuthData = Field(..., description="Authentication data")
    scope_data: ScopeData = Field(..., description="Scope-specific data")
    metadata: IntegrationMetadata = Field(..., description="Integration metadata")
    unique_identifier: str = Field(
        ..., description="Unique identifier for the integration"
    )
    created_by: str = Field(..., description="User creating the integration")


class IntegrationUpdateRequest(BaseModel):
    """Request to update an existing integration - currently only allows name updates"""

    name: str = Field(..., description="Integration name", min_length=1, max_length=255)

    class Config:
        use_enum_values = True


class IntegrationResponse(BaseModel):
    """Response containing integration data"""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Integration] = Field(
        None, description="Integration data if successful"
    )
    error: Optional[str] = Field(None, description="Error message if operation failed")


class IntegrationListResponse(BaseModel):
    """Response containing list of integrations"""

    success: bool = Field(..., description="Whether the operation was successful")
    count: int = Field(..., description="Number of integrations returned")
    integrations: Dict[str, Integration] = Field(
        ..., description="Dictionary of integrations"
    )
    error: Optional[str] = Field(None, description="Error message if operation failed")


# OAuth-related schemas
class OAuthInitiateRequest(BaseModel):
    """Request to initiate OAuth flow"""

    redirect_uri: str = Field(..., description="OAuth redirect URI")
    state: Optional[str] = Field(
        None, description="Optional state parameter for security"
    )


class OAuthCallbackRequest(BaseModel):
    """Request for OAuth callback handling"""

    user_id: str = Field(..., description="User ID to associate with OAuth tokens")


class OAuthTokenResponse(BaseModel):
    """Response containing OAuth token information"""

    user_id: str
    has_valid_token: bool
    token_type: Optional[str] = None
    scope: Optional[str] = None
    expires_at: Optional[float] = None


class OAuthStatusResponse(BaseModel):
    """Response for OAuth status check"""

    status: str
    message: str
    user_id: str


# Sentry-specific schemas
class SentryIntegrationStatus(BaseModel):
    """Sentry integration status for a user"""

    user_id: str
    is_connected: bool
    connected_at: Optional[datetime] = None
    scope: Optional[str] = None
    expires_at: Optional[datetime] = None


class SentryUser(BaseModel):
    """Sentry user information from OAuth response"""

    id: str = Field(..., description="Sentry user ID")
    name: str = Field(..., description="User name")
    email: str = Field(..., description="User email")


class SentryOrganization(BaseModel):
    """Sentry organization information from OAuth response"""

    id: str = Field(..., description="Organization ID")
    slug: str = Field(..., description="Organization slug")
    name: str = Field(..., description="Organization name")


class SentrySaveRequest(BaseModel):
    """Request to save Sentry integration with authorization code"""

    # OAuth authorization code (backend will exchange for tokens)
    code: str = Field(..., description="OAuth authorization code from Sentry")

    # Redirect URI used during OAuth flow (must match what was used for authorization)
    redirect_uri: str = Field(
        ..., description="OAuth redirect URI used during authorization"
    )

    # Integration metadata
    instance_name: str = Field(
        ..., description="User-defined name for this integration instance"
    )
    integration_type: str = Field(default="sentry", description="Type of integration")
    timestamp: str = Field(
        ..., description="ISO timestamp when the integration was created"
    )


class SentrySaveResponse(BaseModel):
    """Response after successfully saving Sentry integration"""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Dict[str, Any]] = Field(
        None, description="Integration data if successful"
    )
    error: Optional[str] = Field(None, description="Error message if operation failed")


# Linear-specific schemas
class LinearIntegrationStatus(BaseModel):
    """Linear integration status for a user"""

    user_id: str
    is_connected: bool
    connected_at: Optional[datetime] = None
    scope: Optional[str] = None
    expires_at: Optional[datetime] = None


class LinearUser(BaseModel):
    """Linear user information from OAuth response"""

    id: str = Field(..., description="Linear user ID")
    name: str = Field(..., description="User name")
    email: str = Field(..., description="User email")


class LinearSaveRequest(BaseModel):
    """Request to save Linear integration with authorization code"""

    # OAuth authorization code (backend will exchange for tokens)
    code: str = Field(..., description="OAuth authorization code from Linear")

    # Redirect URI used during OAuth flow (must match what was used for authorization)
    redirect_uri: str = Field(
        ..., description="OAuth redirect URI used during authorization"
    )

    # Integration metadata
    instance_name: str = Field(
        ..., description="User-defined name for this integration instance"
    )
    integration_type: str = Field(default="linear", description="Type of integration")
    timestamp: str = Field(
        ..., description="ISO timestamp when the integration was created"
    )


class LinearSaveResponse(BaseModel):
    """Response after successfully saving Linear integration"""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Dict[str, Any]] = Field(
        None, description="Integration data if successful"
    )
    error: Optional[str] = Field(None, description="Error message if operation failed")


# Generic save integration schema
class IntegrationSaveRequest(BaseModel):
    """Request to save an integration with configurable and optional fields"""

    # Required fields
    name: str = Field(..., description="Integration name", min_length=1, max_length=255)
    integration_type: IntegrationType = Field(..., description="Type of integration")

    # Optional fields with defaults
    status: IntegrationStatus = Field(
        default=IntegrationStatus.ACTIVE, description="Integration status"
    )
    active: bool = Field(default=True, description="Whether integration is active")

    # Optional auth data with defaults
    auth_data: Optional[AuthData] = Field(
        default_factory=lambda: AuthData(
            access_token=None,
            refresh_token=None,
            token_type="Bearer",
            expires_at=None,
            scope=None,
            code=None,
        ),
        description="Authentication data",
    )

    # Optional scope data with defaults
    scope_data: Optional[ScopeData] = Field(
        default_factory=lambda: ScopeData(
            org_slug=None,
            installation_id=None,
            workspace_id=None,
            project_id=None,
        ),
        description="Scope-specific data",
    )

    # Optional metadata with defaults
    metadata: Optional[IntegrationMetadata] = Field(
        default_factory=lambda: IntegrationMetadata(
            instance_name="",
            created_via="manual",
            description=None,
            version=None,
            tags=[],
        ),
        description="Integration metadata",
    )

    # Optional unique identifier (will be generated if not provided)
    unique_identifier: Optional[str] = Field(
        None,
        description="Unique identifier for the integration (auto-generated if not provided)",
    )


class IntegrationSaveResponse(BaseModel):
    """Response after successfully saving an integration"""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Dict[str, Any]] = Field(
        None, description="Integration data if successful"
    )
    error: Optional[str] = Field(None, description="Error message if operation failed")
