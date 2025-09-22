from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from .sentry_oauth_v2 import SentryOAuthV2
from .linear_oauth import LinearOAuth
from .integrations_schema import (
    SentryIntegrationStatus,
    SentrySaveRequest,
    LinearIntegrationStatus,
    LinearSaveRequest,
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
import logging
import time
import uuid
from datetime import datetime, timedelta


class IntegrationsService:
    """Service layer for integrations"""

    def __init__(self, db: Session):
        self.db = db
        self.config = Config()
        self.sentry_oauth = SentryOAuthV2(self.config)
        self.linear_oauth = LinearOAuth(self.config)

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
                else datetime.utcnow()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data["updated_at"]
                else datetime.utcnow()
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
                    logging.error(
                        f"Token refresh failed: {response.status_code} - {response.text}"
                    )
                    raise Exception(f"Token refresh failed: {response.status_code}")

                token_response = response.json()
                logging.info(
                    f"Token refresh successful, received: {list(token_response.keys())}"
                )

                # Parse token expiration
                expires_at = datetime.utcnow() + timedelta(
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
                setattr(db_integration, "updated_at", datetime.utcnow())

                self.db.commit()
                self.db.refresh(db_integration)

                logging.info(
                    f"Integration {integration_id} tokens refreshed successfully"
                )

                return {
                    "success": True,
                    "access_token": token_response["access_token"],
                    "expires_at": expires_at.isoformat(),
                    "scope": token_response.get("scope", ""),
                }

        except Exception as e:
            logging.error(f"Failed to refresh Sentry token: {str(e)}")
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
                    if datetime.utcnow() >= expires_at:
                        # Token expired, refresh it
                        logging.info(
                            f"Token expired for integration {integration_id}, refreshing..."
                        )
                        refresh_result = await self.refresh_sentry_token(integration_id)
                        return refresh_result["access_token"]
                except ValueError:
                    logging.warning(
                        f"Invalid expiration date format for integration {integration_id}"
                    )

            # Token is still valid, decrypt and return it
            return decrypt_token(auth_data["access_token"])

        except Exception as e:
            logging.error(f"Failed to get valid Sentry token: {str(e)}")
            raise Exception(f"Failed to get valid token: {str(e)}")

    async def _exchange_code_for_tokens(
        self, code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange OAuth authorization code for access tokens and get organization info"""
        try:
            import httpx
            import time

            # Get OAuth client credentials from config
            client_id = self.config("SENTRY_CLIENT_ID", default="")
            client_secret = self.config("SENTRY_CLIENT_SECRET", default="")
            # Use the redirect_uri from the request instead of hardcoded config

            # Comprehensive debugging for OAuth token exchange
            logging.info("=== COMPREHENSIVE OAuth Token Exchange Debug ===")
            logging.info(f"Timestamp: {time.time()}")
            logging.info(f"Code length: {len(code)}")
            logging.info(f"Code preview: {code[:20]}...")
            logging.info(f"Code full: {code}")
            logging.info(f"Redirect URI: {redirect_uri}")
            logging.info(f"Redirect URI length: {len(redirect_uri)}")
            logging.info(f"Client ID configured: {bool(client_id)}")
            logging.info(f"Client ID length: {len(client_id) if client_id else 0}")
            logging.info(
                f"Client ID preview: {client_id[:8] + '...' if client_id and len(client_id) > 8 else client_id}"
            )
            logging.info(f"Client Secret configured: {bool(client_secret)}")
            logging.info(
                f"Client Secret length: {len(client_secret) if client_secret else 0}"
            )
            logging.info(
                f"Client Secret preview: {client_secret[:8] + '...' if client_secret and len(client_secret) > 8 else client_secret}"
            )

            # Log the exact request that will be sent
            logging.info("=== REQUEST DETAILS ===")
            logging.info("Token URL: https://sentry.io/oauth/token/")
            logging.info("Request method: POST")
            logging.info("Content-Type: application/x-www-form-urlencoded")
            logging.info(
                "Request body fields: client_id, client_secret, grant_type, code, redirect_uri"
            )
            logging.info("Grant type: authorization_code")
            logging.info(f"Code: {code}")
            logging.info(f"Redirect URI: {redirect_uri}")
            logging.info(f"Client ID: {client_id}")
            logging.info(
                f"Client Secret: {'*' * len(client_secret) if client_secret else 'NOT_SET'}"
            )
            logging.info("Note: Including redirect_uri in token exchange request")

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

            logging.info(
                f"Exchanging OAuth code for tokens with client_id: {client_id[:8]}..."
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

            logging.info(f"Token exchange request data: {list(token_data.keys())}")
            logging.info(f"Token URL: {token_url}")
            logging.info("Note: Including redirect_uri for validation")

            # Make the token exchange request
            async with httpx.AsyncClient(timeout=30.0) as client:
                logging.info("Making OAuth token exchange request...")

                # Log the exact request data being sent (without secrets)
                debug_data = {
                    k: v for k, v in token_data.items() if k != "client_secret"
                }
                debug_data["client_secret"] = "***REDACTED***"
                logging.info(f"Request payload (debug): {debug_data}")

                # Use form-encoded data as required by OAuth 2.0 spec
                response = await client.post(
                    token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                # Comprehensive response debugging
                logging.info("=== RESPONSE ANALYSIS ===")
                logging.info(f"Response status code: {response.status_code}")
                logging.info(f"Response reason: {response.reason_phrase}")
                logging.info(f"Response URL: {response.url}")
                logging.info(f"Response headers count: {len(response.headers)}")

                # Log all response headers
                for header_name, header_value in response.headers.items():
                    logging.info(f"Response header - {header_name}: {header_value}")

                logging.info(f"Response content length: {len(response.content)}")
                logging.info(f"Response text: {response.text}")
                logging.info(f"Response content (bytes): {response.content}")

                # Try to parse as JSON
                try:
                    response_json = response.json()
                    logging.info(f"Response JSON: {response_json}")
                except Exception as json_error:
                    logging.info(f"Response is not valid JSON: {json_error}")

                if response.status_code != 200:
                    logging.error("=== ERROR ANALYSIS ===")
                    logging.error(
                        f"HTTP Error: {response.status_code} {response.reason_phrase}"
                    )

                    # Try to parse error response for more details
                    try:
                        error_data = response.json()
                        logging.error(f"Error response JSON: {error_data}")

                        error_type = error_data.get("error", "unknown")
                        error_description = error_data.get(
                            "error_description", "No description provided"
                        )

                        logging.error(f"Error type: {error_type}")
                        logging.error(f"Error description: {error_description}")

                        # Detailed analysis based on error type
                        if error_type == "invalid_grant":
                            logging.error("=== INVALID_GRANT ANALYSIS ===")
                            logging.error("Possible causes:")
                            logging.error("1. Authorization code expired (10 minutes)")
                            logging.error("2. Authorization code already used")
                            logging.error("3. Redirect URI mismatch")
                            logging.error("4. Client credentials incorrect")
                            logging.error("5. Grant type incorrect")
                            logging.error("6. Code parameter malformed")

                            # Additional debugging for invalid_grant
                            logging.error("=== DEBUGGING INVALID_GRANT ===")
                            logging.error(f"Code used: {code}")
                            logging.error(f"Code length: {len(code)}")
                            logging.error(f"Client ID used: {client_id}")
                            logging.error("Grant type used: authorization_code")
                            logging.error(
                                "Note: redirect_uri NOT sent in token exchange (per Sentry docs)"
                            )

                    except Exception as parse_error:
                        logging.error(
                            f"Could not parse error response as JSON: {parse_error}"
                        )
                        logging.error(f"Raw response: {response.text}")
                else:
                    logging.info("=== SUCCESS ===")
                    logging.info("OAuth token exchange successful!")

                response.raise_for_status()

                # Parse the response
                token_response = response.json()
                logging.info(
                    f"OAuth token exchange successful, received: {list(token_response.keys())}"
                )

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

                logging.info(
                    f"Successfully exchanged code for tokens: {tokens.get('token_type')} token"
                )
                return tokens

        except Exception as e:
            logging.error(f"Failed to exchange OAuth code for tokens: {str(e)}")
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
                    logging.error(
                        f"Failed to get organization info: {response.status_code}"
                    )
                    return None

                organizations = response.json()
                # Return the first organization (Sentry OAuth is typically scoped to one organization)
                if organizations:
                    org = organizations[0]
                    logging.info(f"Retrieved organization info: {org.get('slug')}")
                    return {
                        "id": str(org.get("id")),
                        "slug": org.get("slug"),
                        "name": org.get("name"),
                    }

                return None

        except Exception as e:
            logging.error(f"Error getting organization info: {str(e)}")
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
            logging.error(f"Error making Sentry API call: {str(e)}")
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
        self, request: SentrySaveRequest
    ) -> Dict[str, Any]:
        """Save Sentry integration with authorization code (backend handles token exchange)"""
        try:
            from .token_encryption import encrypt_token
            import time

            logging.info("=== Sentry Integration Save Debug ===")
            logging.info("Processing Sentry integration with authorization code")
            logging.info(
                f"Request data: instance_name={request.instance_name}, integration_type={request.integration_type}"
            )
            logging.info(f"Authorization code length: {len(request.code)}")
            logging.info(f"Code preview: {request.code[:20]}...")
            logging.info(f"Redirect URI from request: {request.redirect_uri}")
            logging.info(f"Request timestamp: {request.timestamp}")
            logging.info(f"Current timestamp: {time.time()}")

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

                logging.info(
                    f"Time difference between request and now: {time_diff} seconds"
                )

                if time_diff > 600:  # 10 minutes
                    logging.warning(
                        f"Authorization code might be expired (age: {time_diff} seconds)"
                    )
                    raise Exception(
                        f"Authorization code may be expired (age: {time_diff} seconds)"
                    )

            except ValueError as e:
                logging.warning(f"Could not parse request timestamp: {e}")

            # Exchange authorization code for tokens using clean implementation
            logging.info("Exchanging authorization code for tokens")
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
                created_at = datetime.utcnow()

            # Parse token expiration
            try:
                expires_at = datetime.fromisoformat(
                    tokens["expires_at"].replace("Z", "+00:00")
                )
            except (ValueError, KeyError):
                expires_at = datetime.utcnow() + timedelta(
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

            # Check if this Sentry account is already integrated
            sentry_user_id = tokens.get("user", {}).get("id")
            if sentry_user_id:
                existing_integration = await self.check_existing_sentry_integration(
                    org_info["slug"], sentry_user_id
                )
                if existing_integration:
                    logging.warning(
                        f"Sentry account (org: {org_info['slug']}, user: {sentry_user_id}) is already integrated: {existing_integration['integration_id']}"
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
                f"{org_info['slug']}-{tokens.get('user', {}).get('id', 'unknown')}",
            )
            setattr(
                db_integration, "created_by", tokens.get("user", {}).get("id", "system")
            )  # Use actual user ID from OAuth
            setattr(db_integration, "created_at", created_at)
            setattr(db_integration, "updated_at", created_at)

            # Save to database
            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)
            logging.info(
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
            logging.error(f"Error saving Sentry integration: {str(e)}")

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
                .filter(Integration.active == True)
                .filter(Integration.unique_identifier == organization_id)
                .first()
            )

            if db_integration:
                logging.info(
                    f"Found Linear integration {db_integration.integration_id} "
                    f"by org_id {organization_id}"
                )
                return self._db_to_dict(db_integration)

            logging.warning(
                f"No Linear integration found for organization {organization_id}"
            )
            return None

        except Exception as e:
            logging.error(
                f"Error looking up Linear integration by org ID {organization_id}: {str(e)}"
            )
            return None

    async def get_all_integrations(self) -> Dict[str, Dict[str, Any]]:
        """Get all integrations from database (legacy method)"""
        db_integrations = self.db.query(Integration).all()
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

                logging.info(f"Deleting integration: {integration_details}")

                # Perform the deletion
                self.db.delete(db_integration)
                self.db.commit()

                logging.info(
                    f"Integration successfully deleted from database: {integration_id}"
                )
                return True
            else:
                logging.warning(f"Integration not found for deletion: {integration_id}")
                return False
        except Exception as e:
            logging.error(f"Error deleting integration {integration_id}: {str(e)}")
            self.db.rollback()
            return False

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
                setattr(db_integration, "updated_at", datetime.utcnow())
                self.db.commit()
                logging.info(
                    f"Integration status updated: {integration_id} -> active: {active}"
                )
                return True
            return False
        except Exception as e:
            logging.error(f"Error updating integration status: {str(e)}")
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
            setattr(db_integration, "created_at", datetime.utcnow())
            setattr(db_integration, "updated_at", datetime.utcnow())

            # Save to database
            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)
            logging.info(f"Integration created: {integration_id}")

            # Convert to schema model for response
            integration_schema = self._db_to_schema(db_integration)
            return IntegrationResponse(
                success=True, data=integration_schema, error=None
            )

        except Exception as e:
            logging.error(f"Error creating integration: {str(e)}")
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
            logging.info(
                f"Updating integration name: {integration_id} from '{old_name}' to '{request.name}'"
            )

            # Update only the name field
            setattr(db_integration, "name", request.name)
            setattr(db_integration, "updated_at", datetime.utcnow())

            self.db.commit()
            self.db.refresh(db_integration)

            logging.info(f"Integration name successfully updated: {integration_id}")

            integration_schema = self._db_to_schema(db_integration)
            return IntegrationResponse(
                success=True, data=integration_schema, error=None
            )

        except Exception as e:
            logging.error(f"Error updating integration {integration_id}: {str(e)}")
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
            logging.error(f"Error getting integration: {str(e)}")
            return IntegrationResponse(success=False, data=None, error=str(e))

    async def list_integrations_schema(
        self,
        integration_type: Optional[IntegrationType] = None,
        status: Optional[IntegrationStatus] = None,
        active: Optional[bool] = None,
    ) -> IntegrationListResponse:
        """List integrations using schema models with filtering"""
        try:
            query = self.db.query(Integration)

            # Apply filters
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
            logging.error(f"Error listing integrations: {str(e)}")
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
                logging.warning(
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
            logging.info(
                f"Deleting integration (schema): integration_id='{integration_schema.integration_id}', name='{integration_schema.name}', type='{integration_schema.integration_type}', status='{integration_schema.status}', created_by='{integration_schema.created_by}', created_at='{integration_schema.created_at}'"
            )

            # Delete from database
            self.db.delete(db_integration)
            self.db.commit()

            logging.info(f"Integration successfully deleted (schema): {integration_id}")
            return IntegrationResponse(
                success=True, data=integration_schema, error=None
            )

        except Exception as e:
            logging.error(
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
            logging.error(f"Error logging Linear webhook: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to log Linear webhook: {str(e)}",
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

    async def validate_linear_connection(self, user_id: str) -> bool:
        """Validate if a user has a valid Linear connection"""
        return self.linear_oauth.token_store.is_token_valid(user_id)

    async def get_linear_token_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get Linear token information for a user (for debugging)"""
        return self.linear_oauth.token_store.get_tokens(user_id)

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
                logging.info(
                    f"Found existing Linear integration for organization {org_id}: {existing_integration.integration_id}"
                )
                return self._db_to_dict(existing_integration)

            logging.info(
                f"No existing Linear integration found for organization {org_id}"
            )
            return None

        except Exception as e:
            logging.error(f"Error checking for existing Linear integration: {str(e)}")
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
                logging.info(
                    f"Found existing Sentry integration for org {org_slug}, user {user_id}: {existing_integration.integration_id}"
                )
                return self._db_to_dict(existing_integration)

            logging.info(
                f"No existing Sentry integration found for org {org_slug}, user {user_id}"
            )
            return None

        except Exception as e:
            logging.error(f"Error checking for existing Sentry integration: {str(e)}")
            return None

    async def save_linear_integration(
        self, request: LinearSaveRequest
    ) -> Dict[str, Any]:
        """Save Linear integration with authorization code (backend handles token exchange)"""
        try:
            from .token_encryption import encrypt_token
            import time

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
                logging.warning(f"Could not parse request timestamp: {e}")

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
                created_at = datetime.utcnow()

            # Parse token expiration
            try:
                expires_at = datetime.fromtimestamp(
                    tokens.get("expires_at", time.time() + 3600)
                )
            except (ValueError, KeyError):
                expires_at = datetime.utcnow() + timedelta(
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
                user_info.get("id", "system") if user_info else "system",
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
            logging.error(f"Error saving Linear integration: {str(e)}")

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
            setattr(db_integration, "created_at", datetime.utcnow())
            setattr(db_integration, "updated_at", datetime.utcnow())

            # Save to database
            self.db.add(db_integration)
            self.db.commit()
            self.db.refresh(db_integration)

            logging.info(f"Integration saved successfully: {integration_id}")

            # Return the integration data
            return {
                "integration_id": integration_id,
                "name": request.name,
                "integration_type": request.integration_type.value,
                "status": request.status.value,
                "active": request.active,
                "unique_identifier": unique_identifier,
                "created_by": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "has_auth_data": bool(auth_data.access_token),
                "has_scope_data": bool(scope_data.org_slug or scope_data.workspace_id),
                "metadata": metadata.model_dump(mode="json"),
            }

        except Exception as e:
            logging.error(f"Error saving integration: {str(e)}")
            self.db.rollback()
            raise Exception(f"Failed to save integration: {str(e)}")
