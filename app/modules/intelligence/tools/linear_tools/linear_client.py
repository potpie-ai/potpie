"""
Linear Python SDK - A simple interface for the Linear GraphQL API.
"""

import json
import requests
from typing import Dict, Any, Optional
import os
from sqlalchemy.orm import Session


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
            "Authorization": f"{api_key}",
            "Content-Type": "application/json",
        }

    def execute_query(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the Linear API.

        Args:
            query (str): The GraphQL query or mutation
            variables (Dict[str, Any], optional): Variables for the query

        Returns:
            Dict[str, Any]: The response data

        Raises:
            Exception: If the request fails or returns errors
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(self.API_URL, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

        result = response.json()

        if "errors" in result:
            raise Exception(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}")

        return result["data"]

    def get_issue(self, issue_id: str) -> Dict[str, Any]:
        """
        Fetch an issue by its ID.

        Args:
            issue_id (str): The ID of the issue to fetch

        Returns:
            Dict[str, Any]: The issue data
        """
        query = """
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

        variables = {"id": issue_id}
        result = self.execute_query(query, variables)
        return result["issue"]

    def update_issue(self, issue_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an issue.

        Args:
            issue_id (str): The ID of the issue to update
            input_data (Dict[str, Any]): The update data

        Returns:
            Dict[str, Any]: The updated issue data
        """
        mutation = """
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

        variables = {"id": issue_id, "input": input_data}

        result = self.execute_query(mutation, variables)
        return result["issueUpdate"]

    def comment_create(self, issue_id: str, body: str) -> Dict[str, Any]:
        """
        Add a comment to an issue.

        Args:
            issue_id (str): The ID of the issue to comment on
            body (str): The content of the comment

        Returns:
            Dict[str, Any]: The created comment data
        """
        mutation = """
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

        variables = {"input": {"issueId": issue_id, "body": body}}

        result = self.execute_query(mutation, variables)
        return result["commentCreate"]


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
