"""
Tool for getting users in a Jira project (for assignment purposes).
"""

import asyncio
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class GetJiraProjectUsersInput(BaseModel):
    """Input schema for getting Jira project users."""

    project_key: str = Field(
        description="The project key (e.g., 'PROJ'). This is the prefix you see in issue keys like 'PROJ-123'."
    )
    query: Optional[str] = Field(
        default=None,
        description="Optional search query to filter users by name or email (e.g., 'john' or 'smith')",
    )


class GetJiraProjectUsersTool:
    """Tool for getting assignable users in a Jira project."""

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    description = """Get a list of users who can be assigned to issues in a Jira project.

    Use this tool when you need to:
    - Find a user's account ID for assigning issues
    - Search for users by name or email
    - Discover who can be assigned to tasks in a project
    - Finding users in a particular Jira project

    Returns a list of users with their account IDs, display names, and email addresses.
    You can optionally filter by providing a search query (e.g., a person's name).

    IMPORTANT: Use the 'accountId' field when assigning issues, NOT the display name."""

    async def arun(
        self,
        project_key: str,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get assignable users for a Jira project.

        Args:
            project_key: The project key
            query: Optional search query to filter users

        Returns:
            Dictionary with users list and metadata
        """
        try:
            # Check if user has Jira integration
            has_integration = await check_jira_integration_exists(self.user_id, self.db)
            if not has_integration:
                return {
                    "success": False,
                    "error": "No Jira integration found. Please connect your Jira account in the Integrations page.",
                    "message": "Please head to the Integrations screen and connect your Jira account to use Jira operations.",
                }

            # Get the user-specific Jira client
            client = await get_jira_client_for_user(self.user_id, self.db)

            # Get project users
            result = await client.get_project_users(
                project_key=project_key, query=query
            )

            return {
                "success": True,
                "project_key": result["project_key"],
                "users": result["users"],
                "total": result["total"],
                "message": f"Found {result['total']} assignable user(s) in project {project_key}"
                + (f" matching '{query}'" if query else ""),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get users for project {project_key}: {str(e)}",
            }

    def run(
        self,
        project_key: str,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(project_key, query))


def get_jira_project_users_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Factory function to create a configured GetJiraProjectUsersTool.

    Args:
        db: Database session
        user_id: ID of the user

    Returns:
        A configured StructuredTool for getting Jira project users
    """
    tool_instance = GetJiraProjectUsersTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Jira Project Users",
        description="""Get assignable users in a Jira project.

        Use this to find user account IDs needed for assigning issues.
        You can search by name or email using the optional query parameter.

        Inputs:
        - project_key (str): The project key (e.g., 'PROJ')
        - query (str, optional): Search term to filter users by name or email

        Returns a list of users with their account IDs, which you can use when creating or updating issues.
        IMPORTANT: Use the 'accountId' field when assigning, not the display name.""",
        args_schema=GetJiraProjectUsersInput,
    )
