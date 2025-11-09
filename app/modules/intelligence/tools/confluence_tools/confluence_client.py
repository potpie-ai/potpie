"""
Confluence Client 376 Potpie AI Agent Tools

This module provides a wrapper for Confluence Cloud REST API v2 with OAuth 2.0 (3LO) authentication.
OAuth 2.0 (3LO) requires using api.atlassian.com endpoints with the cloud ID.

API Reference: https://developer.atlassian.com/cloud/confluence/rest/v2/intro/
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import httpx

from app.modules.integrations.integration_model import Integration
from app.modules.integrations.integrations_schema import IntegrationType
from app.modules.integrations.token_encryption import decrypt_token
from datetime import datetime, timezone, timedelta


async def check_confluence_integration_exists(user_id: str, db: Session) -> bool:
    """
    Check if user has an active Confluence integration.

    Args:
        user_id: The user ID
        db: Database session

    Returns:
        True if an active Confluence integration exists, False otherwise
    """
    try:
        integration = (
            db.query(Integration)
            .filter(Integration.integration_type == IntegrationType.CONFLUENCE.value)
            .filter(Integration.created_by == user_id)
            .filter(Integration.active == True)  # noqa: E712
            .order_by(Integration.created_at.desc())
            .first()
        )
        return integration is not None
    except Exception as e:
        logging.error(
            f"Error checking Confluence integration for user {user_id}: {str(e)}"
        )
        return False


class ConfluenceClient:
    """Client for interacting with Confluence Cloud using OAuth 2.0 (3LO) authentication."""

    def __init__(self, server: str, access_token: str, cloud_id: str):
        """
        Initialize the Confluence client.

        Args:
            server: The Confluence server URL (e.g., https://yoursite.atlassian.net)
            access_token: OAuth 2.0 (3LO) access token
            cloud_id: Atlassian cloud ID (site ID) - required for OAuth 2.0 API calls
        """
        self.server = server.rstrip("/")
        self.cloud_id = cloud_id
        self.access_token = access_token

        # OAuth 2.0 (3LO) requires using api.atlassian.com with cloud ID
        # Using Confluence REST API v2
        # See: https://developer.atlassian.com/cloud/confluence/rest/v2/intro/
        self.api_base_url = f"https://api.atlassian.com/ex/confluence/{cloud_id}"

        # Create HTTP client with OAuth 2.0 Bearer token
        self.client = httpx.Client(
            base_url=self.api_base_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                # "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logging.info(
            f"Initialized Confluence OAuth 2.0 client for {server} (cloud_id: {cloud_id})"
        )
        logging.info(f"Confluence decrypted access token: {access_token}")

    def close(self):
        """Close the HTTP client connection."""
        if hasattr(self, "client") and self.client:
            self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_spaces(
        self, limit: int = 25, cursor: Optional[str] = None, type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get list of spaces.

        API: GET /wiki/api/v2/spaces

        Args:
            limit: Maximum number of spaces to return (default 25, max 250)
            cursor: Cursor for pagination
            type: Filter by space type (global, personal)

        Returns:
            Response with spaces list and pagination info
        """
        params = {"limit": min(limit, 250)}
        if cursor:
            params["cursor"] = cursor
        if type:
            params["type"] = type

        logging.info(
            f"Calling Confluence API: GET /wiki/api/v2/spaces with params {params}"
        )
        logging.info(f"Full URL: {self.api_base_url}/wiki/api/v2/spaces")
        logging.info(
            f"Access token length: {len(self.access_token) if self.access_token else 0}"
        )
        response = self.client.get("/wiki/api/v2/spaces", params=params)
        response.raise_for_status()
        return response.json()

    def get_space(self, space_id: str) -> Dict[str, Any]:
        """
        Get space by ID.

        API: GET /wiki/api/v2/spaces/{id}

        Args:
            space_id: The space ID

        Returns:
            Space details
        """
        response = self.client.get(f"/wiki/api/v2/spaces/{space_id}")
        response.raise_for_status()
        return response.json()

    def get_space_pages(
        self,
        space_id: str,
        limit: int = 25,
        cursor: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Get pages in a space.

        API: GET /wiki/api/v2/spaces/{id}/pages

        Args:
            space_id: The space ID
            limit: Maximum number of pages to return
            cursor: Cursor for pagination
            status: Page status (current, archived, deleted, draft, trashed)

        Returns:
            Response with pages list and pagination info
        """
        params = {"limit": min(limit, 250), "status": status}
        if cursor:
            params["cursor"] = cursor

        response = self.client.get(
            f"/wiki/api/v2/spaces/{space_id}/pages", params=params
        )
        response.raise_for_status()
        return response.json()

    def get_page(
        self, page_id: str, body_format: str = "storage", get_draft: bool = False
    ) -> Dict[str, Any]:
        """
        Get page by ID.

        API: GET /wiki/api/v2/pages/{id}

        Args:
            page_id: The page ID
            body_format: Body format to return (storage, atlas_doc_format, view)
            get_draft: Whether to return draft version (default: False)

        Returns:
            Page details with content
        """
        params = {"body-format": body_format}
        if get_draft:
            params["get-draft"] = "true"

        response = self.client.get(f"/wiki/api/v2/pages/{page_id}", params=params)
        response.raise_for_status()
        return response.json()

    def search_pages(
        self,
        cql: str,
        limit: int = 25,
        cursor: Optional[str] = None,
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for content using CQL (Confluence Query Language).

        API: GET /wiki/api/v2/search

        Args:
            cql: CQL query string (e.g., "type=page and space=DEMO")
            limit: Maximum number of results
            cursor: Cursor for pagination
            include_archived: Whether to include archived content

        Returns:
            Search results with pagination info
        """
        params = {"cql": cql, "limit": min(limit, 250)}
        if cursor:
            params["cursor"] = cursor
        if include_archived:
            params["includeArchivedSpaces"] = "true"

        response = self.client.get("/wiki/api/v2/search", params=params)
        response.raise_for_status()
        return response.json()

    def create_page(
        self,
        space_id: str,
        title: str,
        body: str,
        parent_id: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Create a new page.

        API: POST /wiki/api/v2/pages

        Args:
            space_id: The space ID where page will be created
            title: Page title
            body: Page content in storage format (HTML)
            parent_id: Optional parent page ID
            status: Page status (current or draft)

        Returns:
            Created page details
        """
        payload: Dict[str, Any] = {
            "spaceId": space_id,
            "status": status,
            "title": title,
            "body": {"representation": "storage", "value": body},
        }

        if parent_id:
            payload["parentId"] = parent_id

        response = self.client.post("/wiki/api/v2/pages", json=payload)
        response.raise_for_status()
        return response.json()

    def update_page(
        self,
        page_id: str,
        version_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Update an existing page.

        API: PUT /wiki/api/v2/pages/{id}

        Args:
            page_id: The page ID to update
            version_number: Current version number (required for updates)
            title: New page title (optional)
            body: New page content in storage format (optional)
            status: Page status

        Returns:
            Updated page details
        """
        payload: Dict[str, Any] = {
            "id": page_id,
            "status": status,
            "version": {"number": version_number + 1},
        }

        if title:
            payload["title"] = title

        if body:
            payload["body"] = {"representation": "storage", "value": body}

        response = self.client.put(f"/wiki/api/v2/pages/{page_id}", json=payload)
        response.raise_for_status()
        return response.json()

    def add_comment(
        self,
        page_id: str,
        comment: str,
        parent_comment_id: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Add a comment to a page (inline or footer comment).

        API: POST /wiki/api/v2/footer-comments or /wiki/api/v2/inline-comments

        Args:
            page_id: The page ID to comment on
            comment: Comment text in storage format (HTML)
            parent_comment_id: Optional parent comment ID for replies
            status: Comment status

        Returns:
            Created comment details
        """
        # Using footer comments (page-level comments)
        payload: Dict[str, Any] = {
            "pageId": page_id,
            "status": status,
            "body": {"representation": "storage", "value": comment},
        }

        if parent_comment_id:
            payload["parentCommentId"] = parent_comment_id

        response = self.client.post("/wiki/api/v2/footer-comments", json=payload)
        response.raise_for_status()
        return response.json()


async def get_confluence_client_for_user(
    user_id: str, db: Session
) -> Optional[ConfluenceClient]:
    """
    Get an authenticated Confluence client for a user.

    Args:
        user_id: The user ID
        db: Database session

    Returns:
        ConfluenceClient instance or None if no integration found
    """
    try:
        # Get user's Confluence integration
        integration = (
            db.query(Integration)
            .filter(Integration.integration_type == IntegrationType.CONFLUENCE.value)
            .filter(Integration.created_by == user_id)
            .filter(Integration.active == True)  # noqa: E712
            .order_by(Integration.created_at.desc())
            .first()
        )

        if not integration:
            logging.warning(f"No Confluence integration found for user {user_id}")
            return None

        # Extract metadata
        metadata = getattr(integration, "integration_metadata", {}) or {}
        site_url = metadata.get("site_url", "")
        cloud_id = metadata.get("site_id", "")

        if not site_url or not cloud_id:
            logging.error("Confluence integration missing site_url or site_id")
            return None

        # Extract and decrypt access token
        auth_data = getattr(integration, "auth_data", {}) or {}
        logging.info(f"Auth data keys for user {user_id}: {list(auth_data.keys())}")
        encrypted_token = auth_data.get("access_token")

        if not encrypted_token:
            logging.error("Confluence integration missing access token")
            logging.error(f"Available auth_data keys: {list(auth_data.keys())}")
            return None

        logging.info(f"Decrypting access token for user {user_id}")
        access_token = decrypt_token(encrypted_token)
        logging.info(f"Successfully decrypted access token for user {user_id}")

        # Check if token needs refresh
        expires_at = auth_data.get("expires_at")
        if expires_at:
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if isinstance(expires_at, datetime):
                # Add 5-minute buffer
                if expires_at < datetime.now(timezone.utc) + timedelta(minutes=5):
                    logging.info("Confluence access token expired, need to refresh")
                    # Token refresh would be handled by integration service
                    # For now, try with current token
                    pass

        # Create and return client
        return ConfluenceClient(
            server=site_url, access_token=access_token, cloud_id=cloud_id
        )

    except Exception as e:
        logging.error(f"Error creating Confluence client for user {user_id}: {str(e)}")
        return None
