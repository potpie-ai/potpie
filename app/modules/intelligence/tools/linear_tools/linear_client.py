"""
Linear Python SDK - A simple interface for the Linear GraphQL API.
"""

import json
import os
from typing import Any, Dict, Optional

import httpx
from sqlalchemy.orm import Session


# Timeout for Linear GraphQL API (connect, read)
LINEAR_REQUEST_TIMEOUT = httpx.Timeout(10.0, 30.0)


class LinearClient:
    """Client for interacting with the Linear GraphQL API."""

    API_URL = "https://api.linear.app/graphql"

    def __init__(self, api_key: str):
        """
        Initialize the Linear API client.

        Args:
            api_key (str): Your Linear API key
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
        }

    def execute_query(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the Linear API (sync).
        Prefer execute_query_async from async routes.
        """
        with httpx.Client(timeout=LINEAR_REQUEST_TIMEOUT) as client:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables
            response = client.post(
                self.API_URL, headers=self.headers, json=payload
            )
            if response.status_code != 200:
                raise Exception(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            result = response.json()
            if "errors" in result:
                raise Exception(
                    f"GraphQL errors: {json.dumps(result['errors'], indent=2)}"
                )
            return result["data"]

    async def execute_query_async(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the Linear API (async, non-blocking).
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        async with httpx.AsyncClient(timeout=LINEAR_REQUEST_TIMEOUT) as client:
            response = await client.post(
                self.API_URL, headers=self.headers, json=payload
            )
            if response.status_code != 200:
                raise Exception(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            result = response.json()
            if "errors" in result:
                raise Exception(
                    f"GraphQL errors: {json.dumps(result['errors'], indent=2)}"
                )
            return result["data"]

    def get_issue(self, issue_id: str) -> Dict[str, Any]:
        """Fetch an issue by its ID (sync). Prefer get_issue_async from async routes."""
        variables = {"id": issue_id}
        result = self.execute_query(self._get_issue_query(), variables)
        return result["issue"]

    async def get_issue_async(self, issue_id: str) -> Dict[str, Any]:
        """Fetch an issue by its ID (async, non-blocking)."""
        variables = {"id": issue_id}
        result = await self.execute_query_async(self._get_issue_query(), variables)
        return result["issue"]

    @staticmethod
    def _get_issue_query() -> str:
        return """
        query GetIssue($id: String!) {
          issue(id: $id) {
            id
            title
            description
            state {
              id
              name
            }
            assignee {
              id
              name
            }
            team {
              id
              name
            }
            priority
            url
            createdAt
            updatedAt
          }
        }
        """

    def update_issue(self, issue_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an issue (sync). Prefer update_issue_async from async routes."""
        variables = {"id": issue_id, "input": input_data}
        result = self.execute_query(self._update_issue_mutation(), variables)
        return result["issueUpdate"]

    async def update_issue_async(
        self, issue_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an issue (async, non-blocking)."""
        variables = {"id": issue_id, "input": input_data}
        result = await self.execute_query_async(
            self._update_issue_mutation(), variables
        )
        return result["issueUpdate"]

    @staticmethod
    def _update_issue_mutation() -> str:
        return """
        mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) {
            success
            issue {
              id
              title
              description
              state {
                id
                name
              }
              assignee {
                id
                name
              }
              priority
              updatedAt
            }
          }
        }
        """

    def comment_create(self, issue_id: str, body: str) -> Dict[str, Any]:
        """Add a comment to an issue (sync). Prefer comment_create_async from async routes."""
        variables = {"input": {"issueId": issue_id, "body": body}}
        result = self.execute_query(self._comment_create_mutation(), variables)
        return result["commentCreate"]

    async def comment_create_async(self, issue_id: str, body: str) -> Dict[str, Any]:
        """Add a comment to an issue (async, non-blocking)."""
        variables = {"input": {"issueId": issue_id, "body": body}}
        result = await self.execute_query_async(
            self._comment_create_mutation(), variables
        )
        return result["commentCreate"]

    @staticmethod
    def _comment_create_mutation() -> str:
        return """
        mutation CreateComment($input: CommentCreateInput!) {
          commentCreate(input: $input) {
            success
            comment {
              id
              body
              createdAt
              user {
                id
                name
              }
            }
          }
        }
        """


class LinearClientConfig:
    """Configuration manager for Linear clients."""

    _instance: Optional["LinearClientConfig"] = None
    _default_client: Optional[LinearClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LinearClientConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Do not initialize client here - will be done on demand
        self._default_client = None

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        return os.getenv("LINEAR_API_KEY")

    async def _get_api_key_from_secrets(
        self, user_id: str, db: Session
    ) -> Optional[str]:
        """Get API key from the secret manager for a specific user."""
        from app.modules.key_management.secret_manager import SecretStorageHandler

        try:
            # Attempt to retrieve the secret for the user
            secret = SecretStorageHandler.get_secret(
                service="linear", customer_id=user_id, service_type="integration", db=db
            )
            return secret
        except Exception:
            # If any error occurs (like 404 Not Found), return None
            return None

    async def get_client(self, user_id: str, db: Session) -> LinearClient:
        """
        Get a Linear client for a specific user.

        Args:
            user_id: The user ID to look up their Linear API key
            db: The database session for secret retrieval

        Returns:
            A configured LinearClient instance

        Raises:
            ValueError: If no API key is available
        """
        # Try to get API key from user-specific secrets
        api_key = await self._get_api_key_from_secrets(user_id, db)

        # Fall back to environment variable if needed
        if not api_key:
            api_key = self._get_api_key_from_env()

        if not api_key:
            raise ValueError(
                "No Linear API key available. Please set LINEAR_API_KEY environment variable "
                "or configure it in user preferences via the secret manager."
            )

        # Create a new client with the API key
        return LinearClient(api_key)

    @property
    def default_client(self) -> LinearClient:
        """Get the default client using environment variables."""
        if self._default_client is None:
            api_key = self._get_api_key_from_env()
            if not api_key:
                raise ValueError(
                    "LINEAR_API_KEY environment variable is not set. "
                    "Set this variable or use a user-specific client instead."
                )
            self._default_client = LinearClient(api_key)
        return self._default_client


async def get_linear_client_for_user(user_id: str, db: Session) -> LinearClient:
    """
    Get a Linear client for a specific user, using their stored API key if available.

    Args:
        user_id: The user's ID to look up their Linear API key
        db: Database session for secret retrieval

    Returns:
        LinearClient: Configured client for the user
    """
    config = LinearClientConfig()
    return await config.get_client(user_id, db)


def get_linear_client() -> LinearClient:
    """
    Get the default Linear client using environment variables.

    This is provided for backward compatibility or non-user-specific operations.

    Returns:
        LinearClient: A client configured with the environment variable

    Raises:
        ValueError: If LINEAR_API_KEY environment variable is not set
    """
    return LinearClientConfig().default_client
