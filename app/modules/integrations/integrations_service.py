from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from .sentry_oauth_v2 import SentryOAuthV2
from .linear_oauth import LinearOAuth
from .jira_oauth import JiraOAuth
from .confluence_oauth import ConfluenceOAuth
from .integrations_schema import (
    SentryIntegrationStatus,
    SentrySaveRequest,
    LinearIntegrationStatus,
    LinearSaveRequest,
    JiraSaveRequest,
    JiraIntegrationStatus,
    ConfluenceSaveRequest,
    ConfluenceIntegrationStatus,
    Integration as IntegrationSchema,
    IntegrationCreateRequest,
    IntegrationUpdateRequest,
    IntegrationResponse,
    IntegrationListResponse,
    IntegrationType,
    IntegrationStatus,
    AuthData,
    ScopeData,
    IntegrationMetadata,
    IntegrationSaveRequest,
)
from .integration_model import Integration
from starlette.config import Config
import time
import uuid
from app.modules.utils.logger import setup_logger
from app.modules.integrations import hash_user_id
from datetime import datetime, timedelta, timezone
from .token_encryption import decrypt_token

logger = setup_logger(__name__)


class IntegrationsService:
    """Service layer for integrations"""

    def __init__(self, db: Session):
        self.db = db
        self.config = Config()
        self.sentry_oauth = SentryOAuthV2(self.config)
        self.linear_oauth = LinearOAuth(self.config)
        self.jira_oauth = JiraOAuth(self.config)
        self.confluence_oauth = ConfluenceOAuth(self.config)

    def _db_to_schema(self, db_integration: Integration) -> IntegrationSchema:
        """Convert database model to schema model"""
        # Convert to dict first to avoid linter issues with SQLAlchemy columns
        data = self._db_to_dict(db_integration)

        return IntegrationSchema(
            integration_id=data["integration_id"],
            name=data["name"],
            integration_type=IntegrationType(data["integration_type"]),
            status=IntegrationStatus(data["status"]),
            active=data["active"],
            auth_data=(
                AuthData(**data["auth_data"])
                if data["auth_data"]
                else AuthData(
                    access_token=None,
                    refresh_token=None,
                    token_type="Bearer",
                    expires_at=None,
                    scope=None,
                    code=None,
                )
            ),
            scope_data=(
                ScopeData(**data["scope_data"])
                if data["scope_data"]
                else ScopeData(
                    org_slug=None,
                    installation_id=None,
                    workspace_id=None,
                    project_id=None,
                )
            ),
            metadata=(
                IntegrationMetadata(**data["metadata"])
                if data["metadata"]
                else IntegrationMetadata(
                    instance_name="", version=None, description=None
                )
            ),
            unique_identifier=data["unique_identifier"],
            created_by=data["created_by"],
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data["created_at"]
                else datetime.now(timezone.utc)
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data["updated_at"]
                else datetime.now(timezone.utc)
            ),
        )

    def _db_to_dict(self, db_integration: Integration) -> Dict[str, Any]:
        """Convert database model to dictionary (for legacy methods)"""
        return {
            "integration_id": str(db_integration.integration_id),
            "name": str(db_integration.name),
            "integration_type": str(db_integration.integration_type),
            "status": str(db_integration.status),
            "active": bool(db_integration.active),
            "auth_data": db_integration.auth_data or {},
            "scope_data": db_integration.scope_data or {},
            "metadata": db_integration.integration_metadata or {},
            "unique_identifier": str(db_integration.unique_identifier),
            "created_by": str(db_integration.created_by),
            "created_at": (
                db_integration.created_at.isoformat()
                if hasattr(db_integration, "created_at")
                and getattr(db_integration, "created_at", None)
                else None
            ),
            "updated_at": (
                db_integration.updated_at.isoformat()
                if hasattr(db_integration, "updated_at")
                and getattr(db_integration, "updated_at", None)
                else None
            ),
        }

    async def get_sentry_integration_status(
        self, user_id: str
    ) -> SentryIntegrationStatus:
        """Get Sentry integration status for a user"""
        user_info = self.sentry_oauth.get_user_info(user_id)

        if not user_info:
            return SentryIntegrationStatus(user_id=user_id, is_connected=False)

        return SentryIntegrationStatus(
            user_id=user_id,
            is_connected=True,
            scope=user_info.get("scope"),
            expires_at=user_info.get("expires_at"),
        )

    async def revoke_sentry_integration(self, user_id: str) -> bool:
        """Revoke Sentry integration for a user"""
        return self.sentry_oauth.revoke_access(user_id)

    def get_sentry_oauth_instance(self) -> SentryOAuthV2:
        """Get the Sentry OAuth integration instance"""
        return self.sentry_oauth

    async def validate_sentry_connection(self, user_id: str) -> bool:
        """Validate if a user has a valid Sentry connection"""
        return self.sentry_oauth.token_store.is_token_valid(user_id)

    async def get_sentry_token_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get Sentry token information for a user (for debugging)"""
        return self.sentry_oauth.token_store.get_tokens(user_id)

    async def refresh_sentry_token(self, integration_id: str) -> Dict[str, Any]:
        """Refresh expired Sentry access token"""
        try:
            from .token_encryption import decrypt_token, encrypt_token
            import httpx

            # Get the integration from database
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_id == integration_id)
                .first()
            )

            if not db_integration:
                raise Exception(f"Integration not found: {integration_id}")

            if getattr(db_integration, "integration_type") != "sentry":
                raise Exception(
                    f"Integration {integration_id} is not a Sentry integration"
                )

            # Get the auth data
            auth_data = getattr(db_integration, "auth_data")
            if not auth_data or not auth_data.get("refresh_token"):
                raise Exception(
                    f"No refresh token found for integration {integration_id}"
                )

            # Decrypt the refresh token
            refresh_token = decrypt_token(auth_data["refresh_token"])

            # Get OAuth client credentials from config
            client_id = self.config("SENTRY_CLIENT_ID", default="")
            client_secret = self.config("SENTRY_CLIENT_SECRET", default="")

            if not client_id or not client_secret:
                raise Exception("Sentry OAuth credentials not configured")

            # Sentry OAuth token endpoint
            token_url = "https://sentry.io/oauth/token/"

            # Prepare refresh request
            refresh_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }

            # Make the refresh request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(token_url, data=refresh_data)

                if response.status_code != 200:
                    # Safely parse response to extract structured error fields
                    sanitized_error = None
                    try:
                        error_data = response.json()
                        error_field = error_data.get("error", "")
                        error_description = error_data.get("error_description", "")
                        if error_field or error_description:
                            sanitized_error = f"error: {error_field}"
                            if error_description:
                                sanitized_error += (
                                    f", error_description: {error_description[:200]}"
                                )
                        else:
                            # JSON parsed but no error fields, use truncated response text
                            response_text = response.text or ""
                            sanitized_error = (
                                response_text[:250]
                                if response_text
                                else "No response body"
                            )
                    except Exception:
                        # Fallback to truncated response.text if JSON parsing fails
                        response_text = response.text or ""
                        sanitized_error = (
                            response_text[:250] if response_text else "No response body"
                        )

                    # Log sanitized error at error level
                    logger.error(
                        f"Token refresh failed: {response.status_code} - {sanitized_error}"
                    )
                    # Log full response body at debug level for detailed troubleshooting
                    logger.debug(
                        f"Token refresh full response: status={response.status_code}, body={response.text}"
                    )
                    # Raise exception with minimal message
                    raise Exception(f"Token refresh failed: {response.status_code}")

                token_response = response.json()
                logger.info(
                    f"Token refresh successful, received: {list(token_response.keys())}"
                )

                # Parse token expiration
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=token_response.get("expires_in", 3600)
                )

                # Update the integration with new tokens
                new_auth_data = AuthData(
                    access_token=encrypt_token(token_response["access_token"]),
                    refresh_token=encrypt_token(
                        token_response.get("refresh_token", refresh_token)
                    ),
                    token_type=token_response.get("token_type", "bearer"),
                    expires_at=expires_at,
                    scope=token_response.get("scope", auth_data.get("scope", "")),
                    code=None,
                )

                # Update database
                setattr(
                    db_integration, "auth_data", new_auth_data.model_dump(mode="json")
                )
                setattr(db_integration, "updated_at", datetime.now(timezone.utc))

                self.db.commit()
                self.db.refresh(db_integration)

                logger.info(
                    f"Integration {integration_id} tokens refreshed successfully"
                )

                return {
                    "success": True,
                    "access_token": token_response["access_token"],
                    "expires_at": expires_at.isoformat(),
                    "scope": token_response.get("scope", ""),
                }

        except Exception as e:
            logger.exception(
                "Failed to refresh Sentry token", integration_id=integration_id
            )
            raise Exception(f"Token refresh failed: {str(e)}")

    async def get_valid_sentry_token(self, integration_id: str) -> str:
        """Get valid Sentry access token, refreshing if necessary"""
        try:
            from .token_encryption import decrypt_token

            # Get the integration from database
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_id == integration_id)
                .first()
            )

            if not db_integration:
                raise Exception(f"Integration not found: {integration_id}")

            # Get the auth data
            auth_data = getattr(db_integration, "auth_data")
            if not auth_data or not auth_data.get("access_token"):
                raise Exception(
                    f"No access token found for integration {integration_id}"
                )

            # Check if token is expired
            expires_at_str = auth_data.get("expires_at")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(
                        expires_at_str.replace("Z", "+00:00")
                    )
                    if datetime.now(timezone.utc) >= expires_at:
                        # Token expired, refresh it
                        logger.info(
                            f"Token expired for integration {integration_id}, refreshing..."
                        )
                        refresh_result = await self.refresh_sentry_token(integration_id)
                        return refresh_result["access_token"]
                except ValueError:
                    logger.warning(
                        f"Invalid expiration date format for integration {integration_id}"
                    )

            # Token is still valid, decrypt and return it
            return decrypt_token(auth_data["access_token"])

        except Exception as e:
            logger.exception(
                "Failed to get valid Sentry token", integration_id=integration_id
            )
            raise Exception(f"Failed to get valid token: {str(e)}")

    async def _exchange_code_for_tokens(
        self, code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange OAuth authorization code for access tokens and get organization info"""
        try:
            import httpx

            # Get OAuth client credentials from config
            client_id = self.config("SENTRY_CLIENT_ID", default="")
            client_secret = self.config("SENTRY_CLIENT_SECRET", default="")
            # Use the redirect_uri from the request instead of hardcoded config

            # Debug logging for OAuth token exchange (DEBUG level only)
            logger.debug(
                "OAuth token exchange starting",
                code_length=len(code),
                has_client_id=bool(client_id),
                has_client_secret=bool(client_secret),
                has_redirect_uri=bool(redirect_uri),
            )

            if not client_id or not client_secret or not redirect_uri:
                missing = []
                if not client_id:
                    missing.append("SENTRY_CLIENT_ID")
                if not client_secret:
                    missing.append("SENTRY_CLIENT_SECRET")
                if not redirect_uri:
                    missing.append("SENTRY_REDIRECT_URI")
                raise Exception(
                    f"Sentry OAuth credentials not configured. Missing: {', '.join(missing)}"
                )

            # Sentry OAuth token endpoint
            token_url = "https://sentry.io/oauth/token/"

            # Prepare token data for OAuth flow
            # Note: Including redirect_uri as it may be required for validation
            token_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            }

            logger.debug(
                "Token exchange request prepared",
                token_url=token_url,
                fields=list(token_data.keys()),
            )

            # Make the token exchange request
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug("Making OAuth token exchange request")

                # Use form-encoded data as required by OAuth 2.0 spec
                response = await client.post(
                    token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                # Log response details
                logger.debug(
                    "OAuth token exchange response",
                    status_code=response.status_code,
                    content_length=len(response.content),
                )

                if response.status_code != 200:
                    # Try to parse error response for more details
                    try:
                        error_data = response.json()
                        error_type = error_data.get("error", "unknown")
                        error_description = error_data.get(
                            "error_description", "No description provided"
                        )

                        logger.error(
                            "OAuth token exchange failed",
                            status_code=response.status_code,
                            error_type=error_type,
                            error_description=error_description,
                        )

                        # Log helpful hints for common errors at DEBUG level
                        if error_type == "invalid_grant":
                            logger.debug(
                                "Invalid grant error - common causes: "
                                "expired code, code already used, redirect URI mismatch, "
                                "or incorrect client credentials"
                            )

                    except Exception:
                        logger.error(
                            "OAuth token exchange failed",
                            status_code=response.status_code,
                            response_text=response.text[:200],
                        )  # Truncate response

                response.raise_for_status()

                # Parse the response
                token_response = response.json()
                logger.info("OAuth token exchange successful")

                # Get organization information using the access token
                org_info = await self._get_sentry_organization_info(
                    token_response["access_token"]
                )

                # Extract token information with organization data
                tokens = {
                    "access_token": token_response.get("access_token"),
                    "refresh_token": token_response.get("refresh_token"),
                    "token_type": token_response.get("token_type", "bearer"),
                    "expires_in": token_response.get("expires_in"),
                    "expires_at": token_response.get("expires_at"),
                    "scope": token_response.get("scope"),
                    "user": token_response.get("user"),
                    "organization": org_info,
                }

                logger.debug(
                    "Token exchange complete",
                    token_type=tokens.get("token_type"),
                    has_refresh_token=bool(tokens.get("refresh_token")),
                )
                return tokens

        except Exception as e:
            logger.exception("Failed to exchange OAuth code for tokens")
            raise Exception(f"OAuth token exchange failed: {str(e)}")

    async def _get_sentry_organization_info(
        self, access_token: str
    ) -> Optional[Dict[str, Any]]:
        """Get organization information from Sentry API"""
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            # Get organizations for the user
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://sentry.io/api/0/organizations/", headers=headers
                )

                if response.status_code != 200:
                    logger.error(
                        f"Failed to get organization info: {response.status_code}"
                    )
                    return None

                organizations = response.json()
                # Return the first organization (Sentry OAuth is typically scoped to one organization)
                if organizations:
                    org = organizations[0]
                    logger.debug(
                        "Retrieved organization info", org_slug=org.get("slug")
                    )
                    return {
                        "id": str(org.get("id")),
                        "slug": org.get("slug"),
                        "name": org.get("name"),
                    }

                return None

        except Exception:
            logger.exception("Error getting organization info")
            return None

    async def make_sentry_api_call(
        self,
        integration_id: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make an API call to Sentry using stored integration tokens"""
        try:
            # Get valid access token (will refresh if expired)
            access_token = await self.get_valid_sentry_token(integration_id)

            # Make the API call to Sentry
            import httpx

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            url = f"https://sentry.io/api/0{endpoint}"

            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, json=data)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=headers, json=data)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise Exception(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.exception("Error making Sentry API call")
            raise Exception(f"Failed to make Sentry API call: {str(e)}")

    async def get_sentry_organizations(
        self, integration_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get Sentry organizations for an integration"""
        return await self.make_sentry_api_call(integration_id, "/organizations/")

    async def get_sentry_projects(
        self, integration_id: str, org_slug: str
    ) -> Optional[Dict[str, Any]]:
        """Get Sentry projects for an organization"""
        return await self.make_sentry_api_call(
            integration_id, f"/organizations/{org_slug}/projects/"
        )

    async def get_sentry_issues(
        self, integration_id: str, org_slug: str, project_slug: str
    ) -> Optional[Dict[str, Any]]:
        """Get Sentry issues for a project"""
        return await self.make_sentry_api_call(
            integration_id, f"/projects/{org_slug}/{project_slug}/issues/"
        )

    async def save_sentry_integration(
        self, request: SentrySaveRequest, user_id: str
    ) -> Dict[str, Any]:
        """Save Sentry integration with authorization code (backend handles token exchange)"""
        try:
            from .token_encryption import encrypt_token

            logger.info(
                "Processing Sentry integration",
                instance_name=request.instance_name,
                integration_type=request.integration_type,
            )
            logger.debug(
                "OAuth code validation",
                code_length=len(request.code),
                has_redirect_uri=bool(request.redirect_uri),
            )

            # Validate the authorization code format and timing
            if not request.code or len(request.code) < 20:
                raise Exception("Invalid authorization code format")

            # Validate redirect URI
            if not request.redirect_uri:
                raise Exception("Redirect URI is required for OAuth token exchange")

            # Check if the code might be expired (OAuth codes typically expire in 10 minutes)
            try:
                request_time = datetime.fromisoformat(
                    request.timestamp.replace("Z", "+00:00")
                )
                # Use timezone-aware current time to match the request time
                from datetime import timezone

                current_time = datetime.now(timezone.utc)
                time_diff = (current_time - request_time).total_seconds()

                if time_diff > 600:  # 10 minutes
                    logger.warning(
                        "Authorization code might be expired", age_seconds=time_diff
                    )
                    raise Exception(
                        f"Authorization code may be expired (age: {time_diff} seconds)"
                    )

            except ValueError as e:
                logger.warning(f"Could not parse request timestamp: {e}")

            # Exchange authorization code for tokens
            logger.debug("Exchanging authorization code for tokens")
            tokens = await self.sentry_oauth.exchange_code_for_tokens(
                request.code, request.redirect_uri
            )

            if not tokens.get("access_token"):
                raise Exception("Failed to obtain access token from OAuth exchange")

            if not tokens.get("organization"):
                raise Exception("Failed to retrieve organization information")

            # Generate a unique integration ID
            integration_id = str(uuid.uuid4())

            # Parse the timestamp
            try:
                created_at = datetime.fromisoformat(
                    request.timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                created_at = datetime.now(timezone.utc)

            # Parse token expiration
            try:
                expires_at = datetime.fromisoformat(
                    tokens["expires_at"].replace("Z", "+00:00")
                )
            except (ValueError, KeyError):
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=tokens.get("expires_in", 3600)
                )

            # Create auth data with encrypted tokens
            auth_data = AuthData(
                access_token=encrypt_token(
                    tokens["access_token"]
                ),  # Encrypt for security
                refresh_token=encrypt_token(
                    tokens["refresh_token"]
                ),  # Encrypt for security
                token_type=tokens.get("token_type", "bearer"),
                expires_at=expires_at,
                scope=tokens.get("scope", ""),
                code=None,  # Don't store the code after exchange
            )

            # Create scope data from organization information
            org_info = tokens["organization"]

            # Check if this Potpie user already integrated this Sentry org
            existing_integration = await self.check_existing_sentry_integration(
                org_info["slug"], user_id
            )
            if existing_integration:
                logger.warning(
                    f"Sentry account (org: {org_info['slug']}, user: {hash_user_id(user_id)}) is already integrated: {existing_integration['integration_id']}"
                )
                raise Exception(
                    f"Sentry account is already integrated. "
                    f"Existing integration ID: {existing_integration['integration_id']}. "
                    f"Please delete the existing integration first if you want to reconnect."
                )

            scope_data = ScopeData(
                org_slug=org_info["slug"],
                installation_id=None,  # Not needed for OAuth flow
                workspace_id=None,
                project_id=None,
            )

            # Create metadata with user and organization info
            metadata = IntegrationMetadata(
                instance_name=request.instance_name,
                created_via="oauth",
                description=f"Sentry integration for {org_info['name']}",
                version=None,
            )

            # Create database model
            db_integration = Integration()
            setattr(db_integration, "integration_id", integration_id)
            setattr(db_integration, "name", request.instance_name)
            setattr(db_integration, "integration_type", IntegrationType.SENTRY.value)
            setattr(db_integration, "status", IntegrationStatus.ACTIVE.value)
            setattr(db_integration, "active", True)
            setattr(db_integration, "auth_data", auth_data.model_dump(mode="json"))
            setattr(db_integration, "scope_data", scope_data.model_dump(mode="json"))
            setattr(
                db_integration, "integration_metadata", metadata.model_dump(mode="json")
            )
            setattr(
                db_integration,
                "unique_identifier",
                f"{org_info['slug']}-{user_id}",  # Use Potpie user_id for uniqueness
            )
            setattr(
                db_integration, "created_by", user_id
            )  # Use Potpie user_id, not Sentry OAuth user ID
            setattr(db_integration, "created_at", created_at)
            setattr(db_integration, "updated_at", created_at)

            # Save to database
            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)
            logger.info(
                f"Integration saved to database with encrypted tokens: {integration_id}"
            )

            # Return the integration data
            return {
                "integration_id": integration_id,
                "instance_name": request.instance_name,
                "status": "active",
                "integration_type": request.integration_type,
                "org_slug": org_info["slug"],
                "org_name": org_info["name"],
                "user_email": tokens.get("user", {}).get("email", "unknown"),
                "created_at": created_at.isoformat(),
                "has_tokens": True,
                "requires_oauth": False,
            }

        except Exception as e:
            logger.exception("Error saving Sentry integration")

            # Provide more helpful error messages based on the error type
            error_message = str(e)
            if "invalid_grant" in error_message:
                error_message = (
                    "OAuth authorization failed. This usually means:\n"
                    "1. The authorization code has expired (codes expire in ~10 minutes)\n"
                    "2. The authorization code has already been used\n"
                    "3. The redirect URI doesn't match what was used during authorization\n"
                    "4. The client credentials are incorrect\n\n"
                    "Please try the OAuth flow again with a fresh authorization code."
                )
            elif "expired" in error_message.lower():
                error_message = (
                    "The authorization code has expired. OAuth codes typically expire in 10 minutes. "
                    "Please initiate a new OAuth flow to get a fresh authorization code."
                )

            raise Exception(f"Failed to save Sentry integration: {error_message}")

    async def get_integration_by_id(
        self, integration_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get integration by ID from database (legacy method)"""
        db_integration = (
            self.db.query(Integration)
            .filter(Integration.integration_id == integration_id)
            .first()
        )
        if db_integration:
            return self._db_to_dict(db_integration)
        return None

    async def get_linear_integration_by_org_id(
        self, organization_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get Linear integration by organization ID using unique_identifier"""
        try:
            # Query for integration using org_id as unique_identifier
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_type == "linear")
                .filter(Integration.active == True)  # noqa: E712
                .filter(Integration.unique_identifier == organization_id)
                .first()
            )

            if db_integration:
                logger.info(
                    f"Found Linear integration {db_integration.integration_id} "
                    f"by org_id {organization_id}"
                )
                return self._db_to_dict(db_integration)

            logger.warning(
                f"No Linear integration found for organization {organization_id}"
            )
            return None

        except Exception as e:
            logger.error(
                f"Error looking up Linear integration by org ID {organization_id}: {str(e)}"
            )
            return None

    async def get_integrations_by_user(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all integrations created by a specific user"""
        db_integrations = (
            self.db.query(Integration).filter(Integration.created_by == user_id).all()
        )
        return {
            str(integration.integration_id): self._db_to_dict(integration)
            for integration in db_integrations
        }

    async def get_integrations_by_type(
        self, integration_type: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get all integrations of a specific type (legacy method)"""
        db_integrations = (
            self.db.query(Integration)
            .filter(Integration.integration_type == integration_type)
            .all()
        )
        return {
            str(integration.integration_id): self._db_to_dict(integration)
            for integration in db_integrations
        }

    async def get_integrations_by_org_slug(
        self, org_slug: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get all integrations for a specific organization (legacy method)"""
        # Query using JSONB field
        db_integrations = (
            self.db.query(Integration)
            .filter(Integration.scope_data["org_slug"].astext == org_slug)
            .all()
        )
        return {
            str(integration.integration_id): self._db_to_dict(integration)
            for integration in db_integrations
        }

    async def delete_integration(self, integration_id: str) -> bool:
        """Delete integration from database (legacy method)"""
        try:
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_id == integration_id)
                .first()
            )

            if db_integration:
                # Log integration details before deletion for audit trail
                integration_details = {
                    "integration_id": str(db_integration.integration_id),
                    "name": str(db_integration.name),
                    "type": str(db_integration.integration_type),
                    "status": str(db_integration.status),
                    "created_by": str(db_integration.created_by),
                    "created_at": (
                        db_integration.created_at.isoformat()
                        if hasattr(db_integration, "created_at")
                        and getattr(db_integration, "created_at", None)
                        else None
                    ),
                }

                logger.info(f"Deleting integration: {integration_details}")

                # Clean up Jira webhooks if this is a Jira integration
                if db_integration.integration_type == "jira":
                    await self._cleanup_jira_webhooks(db_integration)

                # Perform the deletion
                self.db.delete(db_integration)
                self.db.commit()

                logger.info(
                    f"Integration successfully deleted from database: {integration_id}"
                )
                return True
            else:
                logger.warning(f"Integration not found for deletion: {integration_id}")
                return False
        except Exception:
            logger.exception(
                "Error deleting integration", integration_id=integration_id
            )
            self.db.rollback()
            return False

    async def _cleanup_jira_webhooks(self, db_integration: Integration) -> None:
        """Clean up all registered Jira webhooks for an integration before deletion"""
        try:
            metadata = db_integration.integration_metadata or {}
            webhooks = metadata.get("webhooks", [])

            if not webhooks:
                logger.info("No webhooks to clean up for Jira integration")
                return

            # Get access token and site_id from auth_data column (not metadata)
            auth_data = db_integration.auth_data or {}
            access_token = auth_data.get("access_token")

            # Get site_id from scope_data or metadata
            scope_data = db_integration.scope_data or {}
            site_id = scope_data.get("org_slug") or metadata.get("site_id")

            if not access_token or not site_id:
                logger.warning(
                    f"Cannot cleanup webhooks: missing access_token or site_id for integration {db_integration.integration_id}"
                )
                logger.debug(
                    f"auth_data: {auth_data}, scope_data: {scope_data}, metadata: {metadata}"
                )
                return

            access_token = decrypt_token(access_token)

            # Delete each webhook
            deleted_count = 0
            failed_count = 0

            for webhook in webhooks:
                webhook_id = webhook.get("id")
                if webhook_id:
                    try:
                        success = await self.jira_oauth.delete_webhook(
                            cloud_id=site_id,
                            access_token=access_token,
                            webhook_id=str(webhook_id),
                        )
                        if success:
                            deleted_count += 1
                            logger.info(
                                f"Deleted Jira webhook {webhook_id} for integration {db_integration.integration_id}"
                            )
                        else:
                            failed_count += 1
                            logger.warning(
                                f"Failed to delete Jira webhook {webhook_id}"
                            )
                    except Exception as e:
                        failed_count += 1
                        logger.error(
                            f"Error deleting Jira webhook {webhook_id}: {str(e)}"
                        )

            logger.info(
                f"Webhook cleanup complete for integration {db_integration.integration_id}: "
                f"{deleted_count} deleted, {failed_count} failed"
            )

        except Exception:
            logger.exception("Error during webhook cleanup")
            # Don't raise - we still want to delete the integration even if webhook cleanup fails

    async def update_integration_status(
        self, integration_id: str, active: bool
    ) -> bool:
        """Update integration active status (legacy method)"""
        try:
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_id == integration_id)
                .first()
            )

            if db_integration:
                # Use setattr to avoid linter issues with SQLAlchemy columns
                setattr(db_integration, "active", active)
                setattr(db_integration, "updated_at", datetime.now(timezone.utc))
                self.db.commit()
                logger.info(
                    f"Integration status updated: {integration_id} -> active: {active}"
                )
                return True
            return False
        except Exception:
            logger.exception("Error updating integration status")
            self.db.rollback()
            return False

    # New methods using proper schema models
    async def create_integration(
        self, request: IntegrationCreateRequest
    ) -> IntegrationResponse:
        """Create a new integration using the schema model"""
        try:
            integration_id = str(uuid.uuid4())

            # Create the database model
            db_integration = Integration()
            setattr(db_integration, "integration_id", integration_id)
            setattr(db_integration, "name", request.name)
            setattr(db_integration, "integration_type", request.integration_type.value)
            setattr(db_integration, "status", IntegrationStatus.ACTIVE.value)
            setattr(db_integration, "active", True)
            setattr(
                db_integration, "auth_data", request.auth_data.model_dump(mode="json")
            )
            setattr(
                db_integration, "scope_data", request.scope_data.model_dump(mode="json")
            )
            setattr(
                db_integration,
                "integration_metadata",
                request.metadata.model_dump(mode="json"),
            )
            setattr(db_integration, "unique_identifier", request.unique_identifier)
            setattr(db_integration, "created_by", request.created_by)
            setattr(db_integration, "created_at", datetime.now(timezone.utc))
            setattr(db_integration, "updated_at", datetime.now(timezone.utc))

            # Save to database
            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)
            logger.info("Integration created", integration_id=integration_id)

            # Convert to schema model for response
            integration_schema = self._db_to_schema(db_integration)
            return IntegrationResponse(
                success=True, data=integration_schema, error=None
            )

        except Exception as e:
            logger.exception("Error creating integration")
            self.db.rollback()
            return IntegrationResponse(success=False, data=None, error=str(e))

    async def update_integration(
        self, integration_id: str, request: IntegrationUpdateRequest
    ) -> IntegrationResponse:
        """Update an existing integration - currently only allows name updates"""
        try:
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_id == integration_id)
                .first()
            )

            if not db_integration:
                return IntegrationResponse(
                    success=False,
                    data=None,
                    error=f"Integration not found: {integration_id}",
                )

            # Log the update attempt
            old_name = str(db_integration.name)
            logger.info(
                f"Updating integration name: {integration_id} from '{old_name}' to '{request.name}'"
            )

            # Update only the name field
            setattr(db_integration, "name", request.name)
            setattr(db_integration, "updated_at", datetime.now(timezone.utc))

            self.db.commit()
            self.db.refresh(db_integration)

            logger.info(
                "Integration name successfully updated", integration_id=integration_id
            )

            integration_schema = self._db_to_schema(db_integration)
            return IntegrationResponse(
                success=True, data=integration_schema, error=None
            )

        except Exception as e:
            logger.exception(
                "Error updating integration", integration_id=integration_id
            )
            self.db.rollback()
            return IntegrationResponse(success=False, data=None, error=str(e))

    async def get_integration_schema(self, integration_id: str) -> IntegrationResponse:
        """Get integration by ID using schema model"""
        try:
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_id == integration_id)
                .first()
            )

            if not db_integration:
                return IntegrationResponse(
                    success=False,
                    data=None,
                    error=f"Integration not found: {integration_id}",
                )

            integration_schema = self._db_to_schema(db_integration)
            return IntegrationResponse(
                success=True, data=integration_schema, error=None
            )

        except Exception as e:
            logger.exception("Error getting integration")
            return IntegrationResponse(success=False, data=None, error=str(e))

    async def list_integrations_schema(
        self,
        integration_type: Optional[IntegrationType] = None,
        status: Optional[IntegrationStatus] = None,
        active: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> IntegrationListResponse:
        """List integrations using schema models with filtering"""
        try:
            query = self.db.query(Integration)

            # Apply filters
            if user_id is not None:
                query = query.filter(Integration.created_by == user_id)
            if integration_type is not None:
                query = query.filter(
                    Integration.integration_type == integration_type.value
                )
            if status is not None:
                query = query.filter(Integration.status == status.value)
            if active is not None:
                query = query.filter(Integration.active == active)

            db_integrations = query.all()

            # Convert to schema models
            integrations = {
                str(integration.integration_id): self._db_to_schema(integration)
                for integration in db_integrations
            }

            return IntegrationListResponse(
                success=True,
                count=len(integrations),
                integrations=integrations,
                error=None,
            )

        except Exception as e:
            logger.exception("Error listing integrations")
            return IntegrationListResponse(
                success=False, count=0, integrations={}, error=str(e)
            )

    async def validate_oauth_configuration(self) -> Dict[str, Any]:
        """Validate OAuth configuration and provide debugging information"""
        try:
            client_id = self.config("SENTRY_CLIENT_ID", default="")
            client_secret = self.config("SENTRY_CLIENT_SECRET", default="")
            redirect_uri = self.config("SENTRY_REDIRECT_URI", default="")

            validation_result = {
                "client_id": {
                    "configured": bool(client_id),
                    "length": len(client_id) if client_id else 0,
                    "preview": (
                        client_id[:8] + "..."
                        if client_id and len(client_id) > 8
                        else client_id
                    ),
                },
                "client_secret": {
                    "configured": bool(client_secret),
                    "length": len(client_secret) if client_secret else 0,
                    "preview": (
                        client_secret[:8] + "..."
                        if client_secret and len(client_secret) > 8
                        else client_secret
                    ),
                },
                "redirect_uri": {
                    "configured": bool(redirect_uri),
                    "value": redirect_uri,
                },
                "validation_errors": [],
            }

            # Check for common configuration issues
            if not client_id:
                validation_result["validation_errors"].append(
                    "SENTRY_CLIENT_ID is not configured"
                )
            elif len(client_id) < 20:
                validation_result["validation_errors"].append(
                    "SENTRY_CLIENT_ID appears to be too short"
                )

            if not client_secret:
                validation_result["validation_errors"].append(
                    "SENTRY_CLIENT_SECRET is not configured"
                )
            elif len(client_secret) < 20:
                validation_result["validation_errors"].append(
                    "SENTRY_CLIENT_SECRET appears to be too short"
                )

            if not redirect_uri:
                validation_result["validation_errors"].append(
                    "SENTRY_REDIRECT_URI is not configured"
                )
            elif not redirect_uri.startswith(("http://", "https://")):
                validation_result["validation_errors"].append(
                    "SENTRY_REDIRECT_URI must be a valid URL"
                )

            validation_result["is_valid"] = (
                len(validation_result["validation_errors"]) == 0
            )

            return validation_result

        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "validation_errors": [f"Configuration validation failed: {str(e)}"],
            }

    async def delete_integration_schema(
        self, integration_id: str
    ) -> IntegrationResponse:
        """Delete integration using schema model"""
        try:
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_id == integration_id)
                .first()
            )

            if not db_integration:
                logger.warning(
                    f"Integration not found for deletion (schema): {integration_id}"
                )
                return IntegrationResponse(
                    success=False,
                    data=None,
                    error=f"Integration not found: {integration_id}",
                )

            # Convert to schema before deletion for audit trail
            integration_schema = self._db_to_schema(db_integration)

            # Log integration details before deletion for audit trail
            logger.info(
                f"Deleting integration (schema): integration_id='{integration_schema.integration_id}', name='{integration_schema.name}', type='{integration_schema.integration_type}', status='{integration_schema.status}', created_by='{integration_schema.created_by}', created_at='{integration_schema.created_at}'"
            )

            # Clean up Jira webhooks if this is a Jira integration
            if db_integration.integration_type == "jira":
                await self._cleanup_jira_webhooks(db_integration)

            # Delete from database
            self.db.delete(db_integration)
            self.db.commit()

            logger.info(
                "Integration successfully deleted (schema)",
                integration_id=integration_id,
            )
            return IntegrationResponse(
                success=True, data=integration_schema, error=None
            )

        except Exception as e:
            logger.error(
                f"Error deleting integration (schema) {integration_id}: {str(e)}"
            )
            self.db.rollback()
            return IntegrationResponse(success=False, data=None, error=str(e))

    async def log_linear_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log Linear webhook data for debugging and processing"""
        try:
            return {
                "status": "success",
                "message": "Linear webhook logged successfully",
                "logged_at": time.time(),
                "webhook_data": webhook_data,
            }

        except Exception as e:
            logger.error(f"Error logging Linear webhook: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to log Linear webhook: {str(e)}",
                "logged_at": time.time(),
            }

    async def log_jira_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log Jira webhook data for debugging and processing"""
        try:
            # Minimal transformation for now; we store raw payload and timestamp
            return {
                "status": "success",
                "message": "Jira webhook logged successfully",
                "logged_at": time.time(),
                "webhook_data": webhook_data,
            }

        except Exception as e:
            logger.error(f"Error logging Jira webhook: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to log Jira webhook: {str(e)}",
                "logged_at": time.time(),
            }

    # Linear OAuth methods
    async def get_linear_integration_status(
        self, user_id: str
    ) -> LinearIntegrationStatus:
        """Get Linear integration status for a user"""
        user_info = self.linear_oauth.get_user_info(user_id)

        if not user_info:
            return LinearIntegrationStatus(user_id=user_id, is_connected=False)

        return LinearIntegrationStatus(
            user_id=user_id,
            is_connected=True,
            scope=user_info.get("scope"),
            expires_at=user_info.get("expires_at"),
        )

    async def revoke_linear_integration(self, user_id: str) -> bool:
        """Revoke Linear integration for a user"""
        return self.linear_oauth.revoke_access(user_id)

    def get_linear_oauth_instance(self) -> LinearOAuth:
        """Get the Linear OAuth integration instance"""
        return self.linear_oauth

    def get_jira_oauth_instance(self) -> JiraOAuth:
        """Get the Jira OAuth integration instance"""
        return self.jira_oauth

    async def validate_linear_connection(self, user_id: str) -> bool:
        """Validate if a user has a valid Linear connection"""
        return self.linear_oauth.token_store.is_token_valid(user_id)

    async def get_linear_token_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get Linear token information for a user (for debugging)"""
        return self.linear_oauth.token_store.get_tokens(user_id)

    async def get_jira_integration_by_site_id(
        self, site_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get Jira integration by site ID (cloud ID) using unique_identifier"""
        try:
            # Query for integration using prefixed site_id as unique_identifier
            unique_id = f"jira-{site_id}"
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_type == "jira")
                .filter(Integration.active == True)  # noqa: E712
                .filter(Integration.unique_identifier == unique_id)
                .first()
            )

            if db_integration:
                logger.info(
                    f"Found Jira integration {db_integration.integration_id} "
                    f"by site_id {site_id}"
                )
                return self._db_to_dict(db_integration)

            logger.warning(f"No Jira integration found for site {site_id}")
            return None

        except Exception as e:
            logger.error(
                f"Error looking up Jira integration by site ID {site_id}: {str(e)}"
            )
            return None

    async def check_existing_linear_integration(
        self, org_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if a Linear organization is already integrated"""
        try:
            # Query for existing integration with this org_id as unique_identifier
            existing_integration = (
                self.db.query(Integration)
                .filter(
                    Integration.integration_type == IntegrationType.LINEAR.value,
                    Integration.unique_identifier == org_id,
                )
                .first()
            )

            if existing_integration:
                logger.info(
                    f"Found existing Linear integration for organization {org_id}: {existing_integration.integration_id}"
                )
                return self._db_to_dict(existing_integration)

            logger.info(
                f"No existing Linear integration found for organization {org_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error checking for existing Linear integration: {str(e)}")
            return None

    async def check_existing_sentry_integration(
        self, org_slug: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if a Sentry account is already integrated"""
        try:
            # Create the unique identifier that would be used for this Sentry user
            unique_identifier = f"{org_slug}-{user_id}"

            # Query for existing integration with this unique identifier
            existing_integration = (
                self.db.query(Integration)
                .filter(
                    Integration.integration_type == IntegrationType.SENTRY.value,
                    Integration.unique_identifier == unique_identifier,
                )
                .first()
            )

            if existing_integration:
                logger.info(
                    f"Found existing Sentry integration for org {org_slug}, user {hash_user_id(user_id)}: {existing_integration.integration_id}"
                )
                return self._db_to_dict(existing_integration)

            logger.info(
                f"No existing Sentry integration found for org {org_slug}, user {hash_user_id(user_id)}"
            )
            return None

        except Exception as e:
            logger.error(f"Error checking for existing Sentry integration: {str(e)}")
            return None

    async def check_existing_jira_integration(
        self, site_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if a Jira site is already integrated"""
        try:
            unique_id = f"jira-{site_id}"
            existing_integration = (
                self.db.query(Integration)
                .filter(
                    Integration.integration_type == IntegrationType.JIRA.value,
                    Integration.unique_identifier == unique_id,
                )
                .first()
            )

            if existing_integration:
                logger.info(
                    f"Found existing Jira integration for site {site_id}: {existing_integration.integration_id}"
                )
                return self._db_to_dict(existing_integration)

            logger.debug("No existing Jira integration found for site", site_id=site_id)
            return None

        except Exception as e:
            logger.error(f"Error checking for existing Jira integration: {str(e)}")
            return None

    async def save_linear_integration(
        self, request: LinearSaveRequest, user_id: str
    ) -> Dict[str, Any]:
        """Save Linear integration with authorization code (backend handles token exchange)"""
        try:
            from .token_encryption import encrypt_token

            # Validate the authorization code format and timing
            if not request.code or len(request.code) < 10:
                raise Exception("Invalid authorization code format")

            # Validate redirect URI
            if not request.redirect_uri:
                raise Exception("Redirect URI is required for OAuth token exchange")

            # Check if the code might be expired (OAuth codes typically expire in 10 minutes)
            try:
                request_time = datetime.fromisoformat(
                    request.timestamp.replace("Z", "+00:00")
                )
                # Use timezone-aware current time to match the request time
                from datetime import timezone

                current_time = datetime.now(timezone.utc)
                time_diff = (current_time - request_time).total_seconds()

                if time_diff > 600:  # 10 minutes
                    raise Exception(
                        f"Authorization code may be expired (age: {time_diff} seconds)"
                    )

            except ValueError as e:
                logger.warning(f"Could not parse request timestamp: {e}")

            # Exchange authorization code for tokens
            tokens = await self.linear_oauth.exchange_code_for_tokens(
                request.code, request.redirect_uri
            )

            if not tokens.get("access_token"):
                raise Exception("Failed to obtain access token from OAuth exchange")

            # Get user information using the access token
            user_info = await self.linear_oauth.get_user_info_from_api(
                tokens["access_token"]
            )

            if not user_info or not user_info.get("id"):
                raise Exception("Failed to retrieve user information from Linear API")

            linear_user_id = user_info.get("id")
            if not linear_user_id:
                raise Exception("Failed to retrieve user ID from Linear API")

            # Get organization information
            org_info = user_info.get("organization", {})
            org_url_key = org_info.get("urlKey") if org_info else None

            # Check if this Linear organization is already integrated
            org_id = org_info.get("id") if org_info else None
            if org_id:
                existing_org_integration = await self.check_existing_linear_integration(
                    org_id
                )
                if existing_org_integration:
                    raise Exception(
                        f"Linear organization is already integrated. "
                        f"Existing integration ID: {existing_org_integration['integration_id']}. "
                        f"Please delete the existing integration first if you want to reconnect."
                    )

            # Note: We only check for organization-level duplicates, not user-level duplicates
            # This allows multiple users from the same organization to create integrations

            # Generate a unique integration ID
            integration_id = str(uuid.uuid4())

            # Parse the timestamp
            try:
                created_at = datetime.fromisoformat(
                    request.timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                created_at = datetime.now(timezone.utc)

            # Parse token expiration
            try:
                expires_at = datetime.fromtimestamp(
                    tokens.get("expires_at", time.time() + 3600)
                )
            except (ValueError, KeyError):
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=tokens.get("expires_in", 3600)
                )

            # Create auth data with encrypted tokens
            auth_data = AuthData(
                access_token=encrypt_token(
                    tokens["access_token"]
                ),  # Encrypt for security
                refresh_token=None,  # Linear doesn't provide refresh tokens in basic OAuth
                token_type=tokens.get("token_type", "Bearer"),
                expires_at=expires_at,
                scope=tokens.get("scope", ""),
                code=None,  # Don't store the code after exchange
            )

            # Create scope data with organization information
            scope_data = ScopeData(
                org_slug=org_url_key,  # Use Linear's urlKey as org identifier
                installation_id=None,
                workspace_id=None,
                project_id=None,
            )

            # Add organization details to scope data for better tracking
            scope_data_dict = scope_data.model_dump(mode="json")
            if org_info:
                scope_data_dict["org_url_key"] = org_url_key
                scope_data_dict["org_name"] = org_info.get("name")
                scope_data_dict["org_id"] = org_info.get("id")

            # Create integration name based on organization
            org_name = (
                org_info.get("name", "Unknown Organization")
                if org_info
                else "Unknown Organization"
            )
            integration_name = f"Linear {org_name}"

            # Create metadata with user info
            metadata = IntegrationMetadata(
                instance_name=integration_name,
                created_via="oauth",
                description=f"Linear integration for {org_name}",
                version=None,
            )

            # Add user email to metadata for duplicate checking
            metadata_dict = metadata.model_dump(mode="json")
            metadata_dict["user_email"] = user_info.get("email", "unknown")
            metadata_dict["user_name"] = user_info.get("name", "Unknown User")

            # Create database model
            db_integration = Integration()
            setattr(db_integration, "integration_id", integration_id)
            setattr(db_integration, "name", integration_name)
            setattr(db_integration, "integration_type", IntegrationType.LINEAR.value)
            setattr(db_integration, "status", IntegrationStatus.ACTIVE.value)
            setattr(db_integration, "active", True)
            setattr(db_integration, "auth_data", auth_data.model_dump(mode="json"))
            setattr(db_integration, "scope_data", scope_data_dict)
            setattr(db_integration, "integration_metadata", metadata_dict)
            # Create unique identifier using org_id from organization info
            unique_identifier = org_id or f"linear-{org_url_key or 'unknown'}"
            setattr(db_integration, "unique_identifier", unique_identifier)
            setattr(
                db_integration,
                "created_by",
                user_id,  # Use Potpie user_id, not Linear OAuth user ID
            )
            setattr(db_integration, "created_at", created_at)
            setattr(db_integration, "updated_at", created_at)

            # Save to database
            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)

            # Return the integration data
            return {
                "integration_id": integration_id,
                "instance_name": integration_name,
                "status": "active",
                "integration_type": request.integration_type,
                "org_name": org_name,
                "org_url_key": org_url_key,
                "user_name": (
                    user_info.get("name", "Unknown") if user_info else "Unknown"
                ),
                "user_email": (
                    user_info.get("email", "unknown") if user_info else "unknown"
                ),
                "created_at": created_at.isoformat(),
                "has_tokens": True,
                "requires_oauth": False,
            }

        except Exception as e:
            logger.error(f"Error saving Linear integration: {str(e)}")

            # Provide more helpful error messages based on the error type
            error_message = str(e)
            if "invalid_grant" in error_message:
                error_message = (
                    "OAuth authorization failed. This usually means:\n"
                    "1. The authorization code has expired (codes expire in ~10 minutes)\n"
                    "2. The authorization code has already been used\n"
                    "3. The redirect URI doesn't match what was used during authorization\n"
                    "4. The client credentials are incorrect\n\n"
                    "Please try the OAuth flow again with a fresh authorization code."
                )
            elif "expired" in error_message.lower():
                error_message = (
                    "The authorization code has expired. OAuth codes typically expire in 10 minutes. "
                    "Please initiate a new OAuth flow to get a fresh authorization code."
                )

            raise Exception(f"Failed to save Linear integration: {error_message}")

    async def save_integration(
        self, request: IntegrationSaveRequest, user_id: str
    ) -> Dict[str, Any]:
        """Save an integration with configurable and optional fields"""
        try:
            # Generate a unique integration ID
            integration_id = str(uuid.uuid4())

            # Generate unique identifier if not provided
            unique_identifier = (
                request.unique_identifier
                or f"{request.integration_type.value}-{integration_id}"
            )

            # Ensure we have valid auth_data, scope_data, and metadata
            auth_data = request.auth_data or AuthData()
            scope_data = request.scope_data or ScopeData()
            metadata = request.metadata or IntegrationMetadata(
                instance_name=request.name
            )

            # Set default metadata instance_name if not provided
            if not metadata.instance_name:
                metadata.instance_name = request.name

            # Create the database model
            db_integration = Integration()
            setattr(db_integration, "integration_id", integration_id)
            setattr(db_integration, "name", request.name)
            setattr(db_integration, "integration_type", request.integration_type.value)
            setattr(db_integration, "status", request.status.value)
            setattr(db_integration, "active", request.active)
            setattr(db_integration, "auth_data", auth_data.model_dump(mode="json"))
            setattr(db_integration, "scope_data", scope_data.model_dump(mode="json"))
            setattr(
                db_integration,
                "integration_metadata",
                metadata.model_dump(mode="json"),
            )
            setattr(db_integration, "unique_identifier", unique_identifier)
            setattr(db_integration, "created_by", user_id)
            setattr(db_integration, "created_at", datetime.now(timezone.utc))
            setattr(db_integration, "updated_at", datetime.now(timezone.utc))

            # Save to database
            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)

            logger.info("Integration saved successfully", integration_id=integration_id)

            # Return the integration data
            return {
                "integration_id": integration_id,
                "name": request.name,
                "integration_type": request.integration_type.value,
                "status": request.status.value,
                "active": request.active,
                "unique_identifier": unique_identifier,
                "created_by": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "has_auth_data": bool(auth_data.access_token),
                "has_scope_data": bool(scope_data.org_slug or scope_data.workspace_id),
                "metadata": metadata.model_dump(mode="json"),
            }

        except Exception as e:
            logger.error(f"Error saving integration: {str(e)}")
            self.db.rollback()
            raise Exception(f"Failed to save integration: {str(e)}")

    async def save_jira_integration(
        self, request: JiraSaveRequest, user_id: str
    ) -> Dict[str, Any]:
        """Save Jira integration with authorization code"""
        try:
            from .token_encryption import encrypt_token

            if not request.code or len(request.code) < 20:
                raise Exception("Invalid authorization code format")

            tokens = await self.jira_oauth.exchange_code_for_tokens(
                request.code, request.redirect_uri
            )

            access_token = tokens.get("access_token")

            if not access_token:
                raise Exception("Failed to obtain access token from OAuth exchange")

            resources = await self.jira_oauth.get_accessible_resources(access_token)
            if not resources:
                raise Exception("No accessible Jira resources returned for this user")

            resource = resources[0]
            site_id = resource.get("id")
            site_name = resource.get("name", "Jira Site")
            site_url = resource.get("url", "")

            if site_id:
                existing_integration = await self.check_existing_jira_integration(
                    site_id
                )
                if existing_integration:
                    raise Exception(
                        "This Jira site is already connected. Please disconnect it before reconnecting."
                    )

            integration_id = str(uuid.uuid4())

            try:
                created_at = datetime.fromisoformat(
                    request.timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                created_at = datetime.now(timezone.utc)

            expires_at = None
            if tokens.get("expires_at"):
                expires_at = datetime.fromtimestamp(
                    tokens["expires_at"], tz=timezone.utc
                )
            else:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=tokens.get("expires_in", 3600)
                )

            encrypted_access_token = encrypt_token(access_token)
            refresh_token = tokens.get("refresh_token")
            encrypted_refresh_token = (
                encrypt_token(refresh_token) if refresh_token else None
            )

            auth_data = AuthData(
                access_token=encrypted_access_token,
                refresh_token=encrypted_refresh_token,
                token_type=tokens.get("token_type", "Bearer"),
                expires_at=expires_at,
                scope=tokens.get("scope", self.jira_oauth.default_scope),
                code=None,
            )

            scope_data = ScopeData(
                org_slug=site_id,
                installation_id=None,
                workspace_id=None,
                project_id=None,
            )

            instance_name = request.instance_name or site_name

            metadata = IntegrationMetadata(
                instance_name=instance_name,
                created_via="oauth",
                description=f"Jira integration for {site_name}",
                version=None,
                tags=["jira"],
            )

            metadata_dict = metadata.model_dump(mode="json")
            if site_url:
                metadata_dict["site_url"] = site_url
            metadata_dict["site_name"] = site_name
            if site_id:
                metadata_dict["site_id"] = site_id

            # Cache tokens in in-memory store for compatibility endpoints
            self.jira_oauth.token_store.store_tokens(
                user_id,
                {
                    "access_token": access_token,
                    "refresh_token": tokens.get("refresh_token"),
                    "scope": tokens.get("scope"),
                    "expires_at": tokens.get("expires_at"),
                },
            )

            # Attempt to register a webhook on the Jira site so Atlassian will push events to us
            try:
                # Determine webhook callback URL from config or fallback to constructed URL
                config = self.config
                webhook_callback = config("JIRA_WEBHOOK_CALLBACK_URL", default=None)

                if not webhook_callback:
                    # Try to build based on API host if configured
                    api_host = config("API_BASE_URL", default=None)
                    if api_host:
                        webhook_callback = (
                            f"{api_host}/api/v1/integrations/jira/webhook"
                        )

                # Only create webhook if we have a callback URL
                if webhook_callback and site_id and access_token:
                    try:
                        logger.info(
                            "Registering Jira webhook for site %s to %s",
                            site_id,
                            webhook_callback,
                        )
                        webhook_resp = await self.jira_oauth.create_webhook(
                            cloud_id=site_id,
                            access_token=access_token,
                            webhook_url=webhook_callback,
                            events=[
                                "jira:issue_created",
                                "jira:issue_updated",
                                "comment_created",
                            ],
                            name=f"Potpie webhook for {instance_name}",
                        )

                        # persist webhook id in metadata if available
                        webhook_id = None
                        if isinstance(webhook_resp, dict):
                            # OAuth webhook response format: {"webhookRegistrationResult": [{"createdWebhookId": 1}]}
                            registration_result = webhook_resp.get(
                                "webhookRegistrationResult", []
                            )
                            if (
                                registration_result
                                and isinstance(registration_result, list)
                                and len(registration_result) > 0
                            ):
                                webhook_id = registration_result[0].get(
                                    "createdWebhookId"
                                )

                            # Fallback to other formats
                            if not webhook_id:
                                if webhook_resp.get("created") and isinstance(
                                    webhook_resp.get("created"), list
                                ):
                                    webhook_id = webhook_resp["created"][0].get("id")
                                else:
                                    webhook_id = webhook_resp.get(
                                        "id"
                                    ) or webhook_resp.get("self")

                        if webhook_id:
                            metadata_dict.setdefault("webhooks", [])
                            metadata_dict["webhooks"].append(
                                {
                                    "id": webhook_id,
                                    "site_id": site_id,
                                    "url": webhook_callback,
                                }
                            )
                            logger.info(
                                f"Stored webhook ID {webhook_id} in integration metadata"
                            )

                    except Exception as wh_exc:
                        logger.warning(
                            "Failed to register Jira webhook for site %s: %s",
                            site_id,
                            str(wh_exc),
                        )
            except Exception as e:
                logger.error(f"Error during Jira webhook registration: {str(e)}")

            db_integration = Integration()
            setattr(db_integration, "integration_id", integration_id)
            setattr(db_integration, "name", instance_name)
            setattr(db_integration, "integration_type", IntegrationType.JIRA.value)
            setattr(db_integration, "status", IntegrationStatus.ACTIVE.value)
            setattr(db_integration, "active", True)
            setattr(db_integration, "auth_data", auth_data.model_dump(mode="json"))
            setattr(db_integration, "scope_data", scope_data.model_dump(mode="json"))
            setattr(db_integration, "integration_metadata", metadata_dict)
            setattr(
                db_integration,
                "unique_identifier",
                f"jira-{site_id}" if site_id else f"jira-{integration_id}",
            )
            setattr(db_integration, "created_by", user_id)
            setattr(db_integration, "created_at", created_at)
            setattr(db_integration, "updated_at", created_at)

            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)

            return {
                "integration_id": integration_id,
                "instance_name": instance_name,
                "status": "active",
                "integration_type": request.integration_type,
                "site_id": site_id,
                "site_name": site_name,
                "site_url": site_url,
                "created_at": created_at.isoformat(),
                "has_tokens": True,
                "requires_oauth": False,
            }

        except Exception as e:
            logger.error(f"Error saving Jira integration: {str(e)}")
            self.db.rollback()
            raise Exception(f"Failed to save Jira integration: {str(e)}")

    async def _get_jira_context(
        self, integration_id: str, auto_refresh: bool = True
    ) -> Dict[str, Any]:
        """Retrieve decrypted token and site information for a Jira integration.

        Args:
            integration_id: The integration ID to fetch context for
            auto_refresh: If True, automatically refresh expired tokens
        """
        db_integration = (
            self.db.query(Integration)
            .filter(Integration.integration_id == integration_id)
            .first()
        )

        if not db_integration:
            raise Exception(f"Integration not found: {integration_id}")

        if db_integration.integration_type != IntegrationType.JIRA.value:
            raise Exception(f"Integration {integration_id} is not a Jira integration")

        auth_data = getattr(db_integration, "auth_data", {}) or {}
        encrypted_token = auth_data.get("access_token")
        encrypted_refresh_token = auth_data.get("refresh_token")

        if not encrypted_token:
            raise Exception("No access token stored for this integration")

        access_token = decrypt_token(encrypted_token)

        # Check if token is expired and refresh if needed
        if auto_refresh:
            expires_at = auth_data.get("expires_at")
            token_expired = False

            if expires_at:
                if isinstance(expires_at, str):
                    try:
                        expires_at = datetime.fromisoformat(
                            expires_at.replace("Z", "+00:00")
                        )
                    except ValueError:
                        expires_at = None

                if expires_at and datetime.now(timezone.utc) >= expires_at:
                    token_expired = True

            if token_expired and encrypted_refresh_token:
                logger.info(
                    f"Access token expired for integration {integration_id}, refreshing..."
                )
                try:
                    refresh_token = decrypt_token(encrypted_refresh_token)
                    new_tokens = await self.jira_oauth.refresh_access_token(
                        refresh_token
                    )

                    # Update tokens in database
                    from .token_encryption import encrypt_token

                    auth_data["access_token"] = encrypt_token(
                        new_tokens["access_token"]
                    )
                    if new_tokens.get("refresh_token"):
                        auth_data["refresh_token"] = encrypt_token(
                            new_tokens["refresh_token"]
                        )
                    auth_data["expires_at"] = datetime.fromtimestamp(
                        new_tokens["expires_at"], tz=timezone.utc
                    ).isoformat()

                    setattr(db_integration, "auth_data", auth_data)
                    setattr(db_integration, "updated_at", datetime.now(timezone.utc))
                    self.db.commit()

                    access_token = new_tokens["access_token"]
                    logger.info(
                        f"Successfully refreshed token for integration {integration_id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to refresh Jira token: {str(e)}")
                    raise Exception(f"Failed to refresh expired token: {str(e)}")

        metadata = getattr(db_integration, "integration_metadata", {}) or {}
        scope_data = getattr(db_integration, "scope_data", {}) or {}

        site_id = metadata.get("site_id") or scope_data.get("org_slug")
        site_url = metadata.get("site_url") or metadata.get("siteUrl")
        site_name = metadata.get("site_name")

        if not site_id:
            raise Exception("Jira site identifier not available for integration")

        return {
            "access_token": access_token,
            "site_id": site_id,
            "site_url": site_url,
            "site_name": site_name,
            "integration": db_integration,
        }

    async def get_jira_accessible_resources(
        self, integration_id: str
    ) -> Dict[str, Any]:
        """Call Atlassian API to list accessible resources."""
        context = await self._get_jira_context(integration_id)
        access_token = context["access_token"]

        try:
            resources = await self.jira_oauth.get_accessible_resources(access_token)
            return {
                "resources": resources,
                "site_id": context.get("site_id"),
                "site_url": context.get("site_url"),
            }
        except Exception as e:
            logger.error(
                f"Failed to fetch Jira accessible resources for {integration_id}: {str(e)}"
            )
            raise

    async def get_jira_projects(
        self, integration_id: str, start_at: int = 0, max_results: int = 50
    ) -> Dict[str, Any]:
        """Fetch Jira projects available to the integration."""
        context = await self._get_jira_context(integration_id)
        access_token = context["access_token"]
        site_id = context["site_id"]

        url = f"{self.jira_oauth.API_BASE_URL}/ex/jira/{site_id}/rest/api/3/project/search"

        params = {"startAt": start_at, "maxResults": max_results}

        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )

        if response.status_code != 200:
            logger.error(
                "Failed to fetch Jira projects (%s): %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Failed to fetch Jira projects: {response.status_code} {response.text}"
            )

        data = response.json()
        data.update(
            {
                "site_id": site_id,
                "site_url": context.get("site_url"),
                "site_name": context.get("site_name"),
            }
        )
        return data

    async def get_jira_project_details(
        self, integration_id: str, project_key_or_id: str
    ) -> Dict[str, Any]:
        """Fetch details for a single Jira project."""
        context = await self._get_jira_context(integration_id)
        access_token = context["access_token"]
        site_id = context["site_id"]

        url = f"{self.jira_oauth.API_BASE_URL}/ex/jira/{site_id}/rest/api/3/project/{project_key_or_id}"

        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )

        if response.status_code != 200:
            logger.error(
                "Failed to fetch Jira project details (%s): %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Failed to fetch Jira project details: {response.status_code}"
            )

        data = response.json()
        data.update(
            {
                "site_id": site_id,
                "site_url": context.get("site_url"),
                "site_name": context.get("site_name"),
            }
        )
        return data

    async def get_jira_integration_status(self, user_id: str) -> JiraIntegrationStatus:
        """Return Jira integration status for a user."""
        try:
            db_integration = (
                self.db.query(Integration)
                .filter(Integration.integration_type == IntegrationType.JIRA.value)
                .filter(Integration.created_by == user_id)
                .filter(Integration.active == True)  # noqa: E712
                .order_by(Integration.created_at.desc())
                .first()
            )

            if not db_integration:
                return JiraIntegrationStatus(user_id=user_id, is_connected=False)

            auth_data = getattr(db_integration, "auth_data", {}) or {}

            expires_at_value = auth_data.get("expires_at")
            expires_at = None
            if expires_at_value:
                if isinstance(expires_at_value, datetime):
                    expires_at = expires_at_value
                else:
                    try:
                        expires_at = datetime.fromisoformat(
                            str(expires_at_value).replace("Z", "+00:00")
                        )
                    except ValueError:
                        expires_at = None

            scope = auth_data.get("scope")

            return JiraIntegrationStatus(
                user_id=user_id,
                is_connected=True,
                connected_at=db_integration.created_at,
                scope=scope,
                expires_at=expires_at,
            )

        except Exception as e:
            logger.error(f"Error fetching Jira integration status: {str(e)}")
            return JiraIntegrationStatus(user_id=user_id, is_connected=False)

    async def deactivate_jira_integrations_for_user(self, user_id: str) -> int:
        """Deactivate Jira integrations created by the user and cleanup their webhooks."""
        try:
            # First, get all active integrations to cleanup webhooks
            integrations_to_deactivate = (
                self.db.query(Integration)
                .filter(Integration.integration_type == IntegrationType.JIRA.value)
                .filter(Integration.created_by == user_id)
                .filter(Integration.active == True)  # noqa: E712
                .all()
            )

            # Cleanup webhooks for each integration
            for db_integration in integrations_to_deactivate:
                try:
                    await self._cleanup_jira_webhooks(db_integration)
                    logger.info(
                        f"Cleaned up webhooks for integration {db_integration.integration_id}"
                    )
                except Exception as webhook_error:
                    logger.error(
                        f"Failed to cleanup webhooks for integration {db_integration.integration_id}: {str(webhook_error)}"
                    )
                    # Continue with deactivation even if webhook cleanup fails

            # Now deactivate the integrations
            updated = (
                self.db.query(Integration)
                .filter(Integration.integration_type == IntegrationType.JIRA.value)
                .filter(Integration.created_by == user_id)
                .filter(Integration.active == True)  # noqa: E712
                .update(
                    {
                        "active": False,
                        "status": IntegrationStatus.INACTIVE.value,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    synchronize_session=False,
                )
            )

            self.db.commit()
            return updated or 0
        except Exception as e:
            logger.error(
                f"Error deactivating Jira integrations for {user_id}: {str(e)}"
            )
            self.db.rollback()
            return 0

    async def save_confluence_integration(
        self, request: ConfluenceSaveRequest, user_id: str
    ) -> Dict[str, Any]:
        """Save Confluence integration with authorization code (associate with user_id)"""
        try:
            from .token_encryption import encrypt_token

            if not request.code or len(request.code) < 20:
                raise Exception("Invalid authorization code format")

            tokens = await self.confluence_oauth.exchange_code_for_tokens(
                request.code, request.redirect_uri
            )

            access_token = tokens.get("access_token")

            if not access_token:
                raise Exception("Failed to obtain access token from OAuth exchange")

            resources = await self.confluence_oauth.get_accessible_resources(
                access_token
            )
            if not resources:
                raise Exception(
                    "No accessible Confluence resources returned for this user"
                )

            resource = resources[0]
            site_id = resource.get("id")
            site_name = resource.get("name", "Confluence Site")
            site_url = resource.get("url", "")

            if site_id:
                existing_integration = await self.check_existing_confluence_integration(
                    site_id
                )
                if existing_integration:
                    raise Exception(
                        "This Confluence site is already connected. Please disconnect it before reconnecting."
                    )

            integration_id = str(uuid.uuid4())

            try:
                created_at = datetime.fromisoformat(
                    request.timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                created_at = datetime.now(timezone.utc)

            expires_at = None
            if tokens.get("expires_at"):
                expires_at = datetime.fromtimestamp(
                    tokens["expires_at"], tz=timezone.utc
                )
            else:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=tokens.get("expires_in", 3600)
                )

            encrypted_access_token = encrypt_token(access_token)
            refresh_token = tokens.get("refresh_token")
            encrypted_refresh_token = (
                encrypt_token(refresh_token) if refresh_token else None
            )

            auth_data = AuthData(
                access_token=encrypted_access_token,
                refresh_token=encrypted_refresh_token,
                token_type=tokens.get("token_type", "Bearer"),
                expires_at=expires_at,
                scope=tokens.get("scope", self.confluence_oauth.default_scope),
                code=None,
            )

            scope_data = ScopeData(
                org_slug=site_id,
                installation_id=None,
                workspace_id=None,
                project_id=None,
            )

            instance_name = request.instance_name or site_name

            metadata = IntegrationMetadata(
                instance_name=instance_name,
                created_via="oauth",
                description=f"Confluence integration for {site_name}",
                version=None,
                tags=["confluence"],
            )

            metadata_dict = metadata.model_dump(mode="json")
            if site_url:
                metadata_dict["site_url"] = site_url
            metadata_dict["site_name"] = site_name
            if site_id:
                metadata_dict["site_id"] = site_id

            # Cache tokens in in-memory store for compatibility endpoints
            self.confluence_oauth.token_store.store_tokens(
                user_id,
                {
                    "access_token": access_token,
                    "refresh_token": tokens.get("refresh_token"),
                    "scope": tokens.get("scope"),
                    "expires_at": tokens.get("expires_at"),
                },
            )

            # Note: Confluence OAuth 2.0 apps cannot register webhooks
            # Webhooks are only available for Atlassian Connect apps
            logger.info(
                "Confluence OAuth 2.0 integration created for site %s (webhooks not available)",
                site_id,
            )

            db_integration = Integration()
            setattr(db_integration, "integration_id", integration_id)
            setattr(db_integration, "name", instance_name)
            setattr(
                db_integration, "integration_type", IntegrationType.CONFLUENCE.value
            )
            setattr(db_integration, "status", IntegrationStatus.ACTIVE.value)
            setattr(db_integration, "active", True)
            setattr(db_integration, "auth_data", auth_data.model_dump(mode="json"))
            setattr(db_integration, "scope_data", scope_data.model_dump(mode="json"))
            setattr(db_integration, "integration_metadata", metadata_dict)
            setattr(
                db_integration,
                "unique_identifier",
                f"confluence-{site_id}" if site_id else f"confluence-{integration_id}",
            )
            setattr(db_integration, "created_by", user_id)
            setattr(db_integration, "created_at", created_at)
            setattr(db_integration, "updated_at", created_at)

            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)

            return {
                "integration_id": integration_id,
                "instance_name": instance_name,
                "status": "active",
                "integration_type": request.integration_type,
                "site_id": site_id,
                "site_name": site_name,
                "site_url": site_url,
                "created_at": created_at.isoformat(),
                "has_tokens": True,
                "requires_oauth": False,
                "scope": tokens.get("scope"),
            }

        except Exception as e:
            logger.error(f"Error saving Confluence integration: {str(e)}")
            self.db.rollback()
            raise

    async def check_existing_confluence_integration(self, site_id: str) -> bool:
        """Check if a Confluence site is already connected"""
        unique_id = f"confluence-{site_id}"
        existing = (
            self.db.query(Integration)
            .filter(Integration.integration_type == IntegrationType.CONFLUENCE.value)
            .filter(Integration.active == True)  # noqa: E712
            .filter(Integration.unique_identifier == unique_id)
            .first()
        )

        return existing is not None

    async def _get_confluence_context(self, integration_id: str) -> Dict[str, Any]:
        """Helper to get Confluence integration context with decrypted tokens"""
        from .token_encryption import decrypt_token

        db_integration = (
            self.db.query(Integration)
            .filter(Integration.integration_id == integration_id)
            .filter(Integration.integration_type == IntegrationType.CONFLUENCE.value)
            .filter(Integration.active == True)  # noqa: E712
            .first()
        )

        if not db_integration:
            raise Exception(f"Confluence integration {integration_id} not found")

        auth_data = getattr(db_integration, "auth_data", {}) or {}
        metadata = getattr(db_integration, "integration_metadata", {}) or {}
        scope_data = getattr(db_integration, "scope_data", {}) or {}

        encrypted_access_token = auth_data.get("access_token")
        if not encrypted_access_token:
            raise Exception("No access token found for Confluence integration")

        access_token = decrypt_token(encrypted_access_token)

        site_id = metadata.get("site_id") or scope_data.get("org_slug")
        if not site_id:
            raise Exception("Confluence site identifier not available for integration")

        return {
            "access_token": access_token,
            "site_id": site_id,
            "site_url": metadata.get("site_url"),
            "site_name": metadata.get("site_name"),
            "integration": db_integration,
        }

    async def get_confluence_accessible_resources(
        self, integration_id: str
    ) -> Dict[str, Any]:
        """Call Atlassian API to list accessible Confluence resources."""
        context = await self._get_confluence_context(integration_id)
        access_token = context["access_token"]

        try:
            resources = await self.confluence_oauth.get_accessible_resources(
                access_token
            )
            return {
                "resources": resources,
                "site_id": context.get("site_id"),
                "site_url": context.get("site_url"),
            }
        except Exception as e:
            logger.error(
                f"Failed to fetch Confluence accessible resources for {integration_id}: {str(e)}"
            )
            raise

    async def get_confluence_spaces(
        self, integration_id: str, start: int = 0, limit: int = 25
    ) -> Dict[str, Any]:
        """Fetch Confluence spaces available to the integration."""
        context = await self._get_confluence_context(integration_id)
        access_token = context["access_token"]
        site_id = context["site_id"]
        if not site_id:
            raise Exception("Confluence site_id missing in context")

        url = f"{self.confluence_oauth.API_BASE_URL}/ex/confluence/{site_id}/wiki/api/v2/spaces"

        params = {"start": start, "limit": limit}

        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )

        if response.status_code != 200:
            logger.error(
                "Failed to fetch Confluence spaces (%s): %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Failed to fetch Confluence spaces: {response.status_code}"
            )

        data = response.json()
        return {
            "spaces": data.get("results", []),
            "start": data.get("start", 0),
            "limit": data.get("limit", limit),
            "size": data.get("size", 0),
            "_links": data.get("_links", {}),
        }

    async def get_confluence_integration_status(
        self, user_id: str
    ) -> ConfluenceIntegrationStatus:
        """Return Confluence integration status for a user."""
        try:
            db_integration = (
                self.db.query(Integration)
                .filter(
                    Integration.integration_type == IntegrationType.CONFLUENCE.value
                )
                .filter(Integration.created_by == user_id)
                .filter(Integration.active == True)  # noqa: E712
                .order_by(Integration.created_at.desc())
                .first()
            )

            if not db_integration:
                return ConfluenceIntegrationStatus(user_id=user_id, is_connected=False)

            auth_data = getattr(db_integration, "auth_data", {}) or {}

            expires_at_value = auth_data.get("expires_at")
            expires_at = None
            if expires_at_value:
                if isinstance(expires_at_value, datetime):
                    expires_at = expires_at_value
                else:
                    try:
                        expires_at = datetime.fromisoformat(
                            str(expires_at_value).replace("Z", "+00:00")
                        )
                    except ValueError:
                        expires_at = None

            scope = auth_data.get("scope")

            return ConfluenceIntegrationStatus(
                user_id=user_id,
                is_connected=True,
                connected_at=db_integration.created_at,
                scope=scope,
                expires_at=expires_at,
            )

        except Exception as e:
            logger.error(f"Error fetching Confluence integration status: {str(e)}")
            return ConfluenceIntegrationStatus(user_id=user_id, is_connected=False)

    async def deactivate_confluence_integrations_for_user(self, user_id: str) -> int:
        """Deactivate Confluence integrations created by the user."""
        try:
            # Note: Confluence OAuth 2.0 apps don't have webhooks to cleanup
            updated = (
                self.db.query(Integration)
                .filter(
                    Integration.integration_type == IntegrationType.CONFLUENCE.value
                )
                .filter(Integration.created_by == user_id)
                .filter(Integration.active == True)  # noqa: E712
                .update(
                    {
                        "active": False,
                        "status": IntegrationStatus.INACTIVE.value,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    synchronize_session=False,
                )
            )

            self.db.commit()
            return updated or 0
        except Exception as e:
            logger.error(
                f"Error deactivating Confluence integrations for {user_id}: {str(e)}"
            )
            self.db.rollback()
            return 0
