"""
Jira Client for Potpie AI Agent Tools

This module provides a wrapper for Jira Cloud REST API with OAuth 2.0 (3LO) authentication.
OAuth 2.0 (3LO) requires using api.atlassian.com endpoints with the cloud ID.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import httpx

from app.modules.integrations.integration_model import Integration
from app.modules.integrations.integrations_schema import IntegrationType
from app.modules.integrations.token_encryption import decrypt_token
from datetime import datetime, timezone, timedelta


async def check_jira_integration_exists(user_id: str, db: Session) -> bool:
    """
    Check if user has an active Jira integration.

    Args:
        user_id: The user ID
        db: Database session

    Returns:
        True if an active Jira integration exists, False otherwise
    """
    try:
        integration = (
            db.query(Integration)
            .filter(Integration.integration_type == IntegrationType.JIRA.value)
            .filter(Integration.created_by == user_id)
            .filter(Integration.active == True)  # noqa: E712
            .order_by(Integration.created_at.desc())
            .first()
        )
        return integration is not None
    except Exception as e:
        logging.error(f"Error checking Jira integration for user {user_id}: {str(e)}")
        return False


class JiraClient:
    """Client for interacting with Jira Cloud using OAuth 2.0 (3LO) authentication."""

    def __init__(self, server: str, access_token: str, cloud_id: str):
        """
        Initialize the Jira client.

        Args:
            server: The Jira server URL (e.g., https://yoursite.atlassian.net) - used for constructing issue URLs
            access_token: OAuth 2.0 (3LO) access token
            cloud_id: Atlassian cloud ID (site ID) - required for OAuth 2.0 API calls
        """
        self.server = server.rstrip("/")
        self.cloud_id = cloud_id
        self.access_token = access_token

        # OAuth 2.0 (3LO) requires using api.atlassian.com with cloud ID
        # See: https://developer.atlassian.com/cloud/jira/platform/oauth-2-3lo-apps/
        self.api_base_url = f"https://api.atlassian.com/ex/jira/{cloud_id}"

        # Create HTTP client with OAuth 2.0 Bearer token
        self.client = httpx.Client(
            base_url=self.api_base_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logging.info(
            f"Initialized Jira OAuth 2.0 client for {server} (cloud_id: {cloud_id})"
        )

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

    @staticmethod
    def _text_to_adf(text: str) -> Dict[str, Any]:
        """
        Convert plain text to Atlassian Document Format (ADF).

        Args:
            text: Plain text string

        Returns:
            ADF formatted document
        """
        if not text:
            return {"type": "doc", "version": 1, "content": []}

        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        content = []

        for para in paragraphs:
            if para.strip():
                # Split by single newlines for inline breaks
                lines = para.split("\n")
                para_content = []

                for i, line in enumerate(lines):
                    if line.strip():
                        para_content.append({"type": "text", "text": line})
                        # Add hard break between lines (except last)
                        if i < len(lines) - 1:
                            para_content.append({"type": "hardBreak"})

                if para_content:
                    content.append({"type": "paragraph", "content": para_content})

        return {
            "type": "doc",
            "version": 1,
            "content": (
                content
                if content
                else [
                    {"type": "paragraph", "content": [{"type": "text", "text": text}]}
                ]
            ),
        }

    @staticmethod
    def _adf_to_text(adf: Any) -> str:
        """
        Convert Atlassian Document Format (ADF) to plain text.

        Args:
            adf: ADF document (dict) or plain string

        Returns:
            Plain text string
        """
        # If it's already a string, return it
        if isinstance(adf, str):
            return adf

        # If it's None or empty, return empty string
        if not adf:
            return ""

        # If it's not a dict, convert to string
        if not isinstance(adf, dict):
            return str(adf)

        # Extract text from ADF structure
        def extract_text(node: Dict[str, Any]) -> str:
            if not isinstance(node, dict):
                return ""

            node_type = node.get("type", "")

            # Text node - return the text content
            if node_type == "text":
                return node.get("text", "")

            # Hard break - return newline
            if node_type == "hardBreak":
                return "\n"

            # Paragraph - process content and add double newline after
            if node_type == "paragraph":
                content = node.get("content", [])
                text = "".join(extract_text(child) for child in content)
                return text + "\n\n" if text else ""

            # Document or other container - process content
            if "content" in node:
                return "".join(extract_text(child) for child in node.get("content", []))

            return ""

        text = extract_text(adf)
        # Remove trailing newlines and return
        return text.rstrip("\n")

    def get_issue(
        self, issue_key: str, fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get a Jira issue by its key.

        Args:
            issue_key: The issue key (e.g., 'PROJ-123')
            fields: Optional list of field names to retrieve

        Returns:
            Dictionary containing issue details
        """
        try:
            params = {}
            if fields:
                params["fields"] = ",".join(fields)
            else:
                # Request common fields by default
                params["fields"] = [
                    "summary",
                    "status",
                    "assignee",
                    "reporter",
                    "priority",
                    "created",
                    "updated",
                    "description",
                    "issuetype",
                    "project",
                    "labels",
                ]

            response = self.client.get(f"/rest/api/3/issue/{issue_key}", params=params)
            response.raise_for_status()
            issue = response.json()
            return self._issue_to_dict(issue)
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to get issue {issue_key}: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to get issue {issue_key}: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to get issue {issue_key}: {str(e)}")
            raise Exception(f"Failed to get issue {issue_key}: {str(e)}")

    def search_issues(
        self,
        jql: str,
        max_results: int = 50,
        next_page_token: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for issues using JQL with the enhanced search API.

        Args:
            jql: JQL query string
            max_results: Maximum number of results to return (default: 50)
            next_page_token: Token for pagination to get the next page of results
            fields: Optional list of field names to retrieve

        Returns:
            Dictionary with search results including:
            - issues: List of issue dictionaries
            - max_results: Maximum results per page
            - is_last: Boolean indicating if this is the last page
            - next_page_token: Token for fetching next page (if not last page)
        """
        try:
            payload = {
                "jql": jql,
                "maxResults": max_results,
            }
            if fields:
                payload["fields"] = fields
            else:
                # Request common fields by default
                payload["fields"] = [
                    "summary",
                    "status",
                    "assignee",
                    "reporter",
                    "priority",
                    "created",
                    "updated",
                    "description",
                    "issuetype",
                    "project",
                    "labels",
                ]

            if next_page_token:
                payload["nextPageToken"] = next_page_token

            # Use the new /search/jql endpoint with token-based pagination
            # See: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issue-search/
            response = self.client.post("/rest/api/3/search/jql", json=payload)
            response.raise_for_status()
            results = response.json()

            logging.info(f"JQL query result: {results}")

            issues = results.get("issues", [])
            is_last = results.get("isLast", True)
            next_token = results.get("nextPageToken")

            result = {
                "max_results": max_results,
                "is_last": is_last,
                "issues": [self._issue_to_dict(issue) for issue in issues],
            }

            # Only include next_page_token if there are more pages
            if not is_last and next_token:
                result["next_page_token"] = next_token

            return result
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to search issues with JQL '{jql}': {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to search issues: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to search issues with JQL '{jql}': {str(e)}")
            raise Exception(f"Failed to search issues: {str(e)}")

    def create_issue(
        self,
        project_key: str,
        summary: str,
        description: str,
        issue_type: str = "Task",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a new Jira issue.

        Args:
            project_key: The project key (e.g., 'PROJ')
            summary: Issue summary/title
            description: Issue description
            issue_type: Issue type (default: 'Task')
            **kwargs: Additional fields (priority, assignee, labels, etc.)

        Returns:
            Dictionary containing the created issue details
        """
        try:
            # Convert description to ADF format for Jira Cloud API v3
            fields = {
                "project": {"key": project_key},
                "summary": summary,
                "description": self._text_to_adf(description),
                "issuetype": {"name": issue_type},
            }

            # Add optional fields
            if "priority" in kwargs:
                fields["priority"] = {"name": kwargs["priority"]}
            if "assignee" in kwargs:
                fields["assignee"] = {"id": kwargs["assignee"]}
            if "labels" in kwargs:
                fields["labels"] = kwargs["labels"]
            if "parent" in kwargs:
                fields["parent"] = {"key": kwargs["parent"]}

            # Add any custom fields
            for key, value in kwargs.items():
                if key not in ["priority", "assignee", "labels", "parent"]:
                    fields[key] = value

            payload = {"fields": fields}
            response = self.client.post("/rest/api/3/issue", json=payload)
            response.raise_for_status()
            issue = response.json()
            issue_key = issue.get("key")
            logging.info(f"Created issue {issue_key}")

            # Fetch the created issue to return full details
            return self.get_issue(issue_key)
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to create issue: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to create issue: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to create issue: {str(e)}")
            raise Exception(f"Failed to create issue: {str(e)}")

    def update_issue(self, issue_key: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing Jira issue.

        Args:
            issue_key: The issue key to update
            fields: Dictionary of fields to update

        Returns:
            Dictionary containing the updated issue details
        """
        try:
            # Convert description to ADF format if it's a plain string
            if "description" in fields and isinstance(fields["description"], str):
                fields["description"] = self._text_to_adf(fields["description"])

            payload = {"fields": fields}
            response = self.client.put(f"/rest/api/3/issue/{issue_key}", json=payload)
            response.raise_for_status()
            logging.info(f"Updated issue {issue_key}")
            # Fetch fresh data after update
            return self.get_issue(issue_key)
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to update issue {issue_key}: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to update issue {issue_key}: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to update issue {issue_key}: {str(e)}")
            raise Exception(f"Failed to update issue {issue_key}: {str(e)}")

    def add_comment(self, issue_key: str, comment_body: str) -> Dict[str, Any]:
        """
        Add a comment to a Jira issue.

        Args:
            issue_key: The issue key to comment on
            comment_body: The comment text

        Returns:
            Dictionary containing comment details
        """
        try:
            # Convert comment to ADF format for Jira Cloud API v3
            payload = {"body": self._text_to_adf(comment_body)}
            response = self.client.post(
                f"/rest/api/3/issue/{issue_key}/comment", json=payload
            )
            response.raise_for_status()
            comment = response.json()
            logging.info(f"Added comment to issue {issue_key}")
            return {
                "id": comment.get("id"),
                "body": comment.get("body"),
                "author": comment.get("author", {}).get("displayName", "Unknown"),
                "created": comment.get("created"),
            }
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to add comment to {issue_key}: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to add comment: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to add comment to {issue_key}: {str(e)}")
            raise Exception(f"Failed to add comment: {str(e)}")

    def transition_issue(self, issue_key: str, transition_name: str) -> Dict[str, Any]:
        """
        Transition an issue to a new status.

        Args:
            issue_key: The issue key to transition
            transition_name: The name of the transition (e.g., 'Done', 'In Progress')

        Returns:
            Dictionary containing the updated issue details
        """
        try:
            transitions = self.get_transitions(issue_key)

            # Find the transition by name
            transition_id = None
            for t in transitions:
                if t.get("name", "").lower() == transition_name.lower():
                    transition_id = t.get("id")
                    break

            if not transition_id:
                available = [t.get("name") for t in transitions]
                raise Exception(
                    f"Transition '{transition_name}' not found. Available transitions: {', '.join(available)}"
                )

            payload = {"transition": {"id": transition_id}}
            response = self.client.post(
                f"/rest/api/3/issue/{issue_key}/transitions", json=payload
            )
            response.raise_for_status()
            logging.info(f"Transitioned issue {issue_key} to {transition_name}")

            # Fetch fresh data after transition
            return self.get_issue(issue_key)
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to transition issue {issue_key}: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to transition issue: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to transition issue {issue_key}: {str(e)}")
            raise Exception(f"Failed to transition issue: {str(e)}")

    def get_transitions(self, issue_key: str) -> List[Dict[str, str]]:
        """
        Get available transitions for an issue.

        Args:
            issue_key: The issue key

        Returns:
            List of available transitions
        """
        try:
            response = self.client.get(f"/rest/api/3/issue/{issue_key}/transitions")
            response.raise_for_status()
            data = response.json()
            transitions = data.get("transitions", [])
            return [{"id": t.get("id"), "name": t.get("name")} for t in transitions]
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to get transitions for {issue_key}: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to get transitions: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to get transitions for {issue_key}: {str(e)}")
            raise Exception(f"Failed to get transitions: {str(e)}")

    def assign_issue(self, issue_key: str, assignee_id: str) -> Dict[str, Any]:
        """
        Assign an issue to a user.

        Args:
            issue_key: The issue key to assign
            assignee_id: The account ID of the user to assign to

        Returns:
            Dictionary containing the updated issue details
        """
        try:
            payload = {"accountId": assignee_id}
            response = self.client.put(
                f"/rest/api/3/issue/{issue_key}/assignee", json=payload
            )
            response.raise_for_status()
            logging.info(f"Assigned issue {issue_key} to user {assignee_id}")

            # Fetch fresh data after assignment
            return self.get_issue(issue_key)
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to assign issue {issue_key}: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to assign issue: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to assign issue {issue_key}: {str(e)}")
            raise Exception(f"Failed to assign issue: {str(e)}")

    def get_projects(self, start_at: int = 0, max_results: int = 50) -> Dict[str, Any]:
        """
        Get all projects accessible to the user.

        Args:
            start_at: Starting index for pagination (default: 0)
            max_results: Maximum number of results to return (default: 50)

        Returns:
            Dictionary containing list of projects with their details
        """
        try:
            params = {
                "startAt": start_at,
                "maxResults": max_results,
            }

            response = self.client.get("/rest/api/3/project/search", params=params)
            response.raise_for_status()
            data = response.json()

            # Extract and format project information
            projects = []
            for project in data.get("values", []):
                projects.append(
                    {
                        "id": project.get("id"),
                        "key": project.get("key"),
                        "name": project.get("name"),
                        "project_type": project.get("projectTypeKey"),
                        "style": project.get("style"),
                        "description": project.get("description", ""),
                        "lead": (
                            project.get("lead", {}).get("displayName")
                            if project.get("lead")
                            else None
                        ),
                        "url": f"{self.server}/browse/{project.get('key')}",
                    }
                )

            total = data.get("total", len(projects))
            is_last = data.get("isLast", (start_at + len(projects)) >= total)

            logging.info(f"Retrieved {len(projects)} projects (start_at={start_at})")

            return {
                "total": total,
                "start_at": start_at,
                "max_results": max_results,
                "is_last": is_last,
                "projects": projects,
            }
        except httpx.HTTPStatusError as e:
            logging.error(
                f"Failed to fetch projects: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"Failed to fetch projects: {e.response.status_code} {e.response.text}"
            )
        except Exception as e:
            logging.error(f"Failed to fetch projects: {str(e)}")
            raise Exception(f"Failed to fetch projects: {str(e)}")

    def _issue_to_dict(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Jira issue dict to a standardized dictionary."""
        try:
            fields = issue.get("fields", {})
            status = fields.get("status", {})
            priority = fields.get("priority", {})
            assignee = fields.get("assignee", {})
            reporter = fields.get("reporter", {})
            issuetype = fields.get("issuetype", {})
            project = fields.get("project", {})

            # Convert ADF description to plain text for easier consumption
            description = fields.get("description")
            if description:
                description = self._adf_to_text(description)

            return {
                "key": issue.get("key"),
                "id": issue.get("id"),
                "summary": fields.get("summary"),
                "description": description,
                "status": status.get("name"),
                "priority": priority.get("name") if priority else None,
                "assignee": assignee.get("displayName") if assignee else None,
                "assignee_id": assignee.get("accountId") if assignee else None,
                "reporter": reporter.get("displayName") if reporter else None,
                "created": fields.get("created"),
                "updated": fields.get("updated"),
                "issue_type": issuetype.get("name"),
                "project": project.get("key"),
                "project_name": project.get("name"),
                "labels": fields.get("labels", []),
                "url": f"{self.server}/browse/{issue.get('key')}",
            }
        except Exception as e:
            logging.error(f"Error converting issue to dict: {str(e)}")
            return {
                "key": issue.get("key", "unknown"),
                "error": f"Failed to parse issue details: {str(e)}",
            }


async def get_jira_client_for_user(user_id: str, db: Session) -> JiraClient:
    """
    Get an authenticated Jira client for a user.

    Args:
        user_id: The user ID
        db: Database session

    Returns:
        An authenticated JiraClient instance

    Raises:
        Exception: If no Jira integration found or authentication fails
    """
    # Query for active Jira integration for the user
    integration = (
        db.query(Integration)
        .filter(Integration.integration_type == IntegrationType.JIRA.value)
        .filter(Integration.created_by == user_id)
        .filter(Integration.active == True)  # noqa: E712
        .order_by(Integration.created_at.desc())
        .first()
    )

    if not integration:
        raise Exception(
            "No Jira integration found. Please connect your Jira account in the Integrations page."
        )

    # Get authentication data - auth_data is a JSONB column
    auth_data = integration.auth_data or {}
    logging.info(
        f"Retrieved Jira integration for user {user_id}, integration_id: {integration.integration_id}"
    )
    logging.debug(f"Auth data keys: {list(auth_data.keys())}")

    encrypted_token = auth_data.get("access_token")
    encrypted_refresh_token = auth_data.get("refresh_token")

    if not encrypted_token:
        logging.error(
            f"No access token in auth_data for integration {integration.integration_id}"
        )
        logging.debug(f"Available keys in auth_data: {list(auth_data.keys())}")
        raise Exception(
            "No access token found for Jira integration. Please reconnect your Jira account."
        )

    # Decrypt token
    try:
        access_token = decrypt_token(encrypted_token)
        logging.debug(f"Successfully decrypted access token for user {user_id}")
    except Exception as e:
        logging.error(f"Failed to decrypt access token: {str(e)}")
        raise Exception(
            "Failed to decrypt Jira access token. Please reconnect your Jira account."
        )

    # Check if token is expired and refresh if needed
    expires_at = auth_data.get("expires_at")
    token_needs_refresh = False

    if expires_at:
        # Handle both string and datetime objects from JSONB
        if isinstance(expires_at, str):
            try:
                expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except ValueError:
                logging.warning(f"Could not parse expires_at: {expires_at}")
                expires_at = None
        elif isinstance(expires_at, datetime):
            # Already a datetime object, ensure it has timezone
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at:
            now = datetime.now(timezone.utc)
            # Refresh if token expires within 5 minutes (300 seconds)
            buffer_time = timedelta(seconds=300)
            logging.info(
                f"Token expires at: {expires_at}, current time: {now}, buffer: 5 minutes"
            )
            if now >= (expires_at - buffer_time):
                token_needs_refresh = True
                if now >= expires_at:
                    logging.warning(
                        f"Access token has already expired for user {user_id}"
                    )
                else:
                    logging.info(
                        f"Access token expires soon for user {user_id}, refreshing proactively"
                    )
    else:
        logging.warning(
            f"No expires_at found for integration {integration.integration_id}, cannot check expiry"
        )

    if token_needs_refresh:
        if not encrypted_refresh_token:
            logging.error(
                f"Token expired but no refresh token available for user {user_id}"
            )
            raise Exception("Access token expired. Please reconnect your Jira account.")

        # Refresh the token
        logging.info(f"Refreshing expired access token for user {user_id}...")
        from app.modules.integrations.integrations_service import IntegrationsService

        service = IntegrationsService(db)
        # Refresh will happen automatically in _get_jira_context
        context = await service._get_jira_context(
            integration.integration_id, auto_refresh=True
        )
        access_token = context["access_token"]
        logging.info(f"Successfully refreshed access token for user {user_id}")

    # Get site information
    metadata = integration.integration_metadata or {}
    scope_data = integration.scope_data or {}

    site_id = metadata.get("site_id") or scope_data.get("org_slug")
    site_url = metadata.get("site_url") or metadata.get("siteUrl")

    if not site_id:
        logging.error(
            f"No site_id found in metadata or scope_data for integration {integration.integration_id}"
        )
        logging.debug(f"Metadata: {metadata}, Scope data: {scope_data}")
        raise Exception(
            "Jira site ID not found in integration. Please reconnect your Jira account."
        )

    if not site_url:
        logging.error(
            f"No site_url found in metadata for integration {integration.integration_id}"
        )
        raise Exception(
            "Jira site URL not found in integration. Please reconnect your Jira account."
        )

    logging.info(f"Creating Jira client for site: {site_url}, site_id: {site_id}")

    # Create and return client
    try:
        return JiraClient(server=site_url, access_token=access_token, cloud_id=site_id)
    except Exception as e:
        logging.error(f"Failed to create Jira client: {str(e)}")
        raise Exception(f"Failed to initialize Jira client: {str(e)}")
