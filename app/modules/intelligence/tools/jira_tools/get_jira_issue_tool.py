"""Tool for fetching Jira issue details."""

from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class GetJiraIssueInput(BaseModel):
    """Input schema for getting a Jira issue."""

    issue_key: str = Field(
        description="The Jira issue key (e.g., 'PROJ-123', 'BUG-456')"
    )


class GetJiraIssueTool:
    """Tool for fetching Jira issue details."""

    name = "Get Jira Issue"
    description = """Fetch detailed information about a Jira issue by its key.
    
    Use this tool when you need to:
    - Get current status, assignee, or priority of an issue
    - Read the description and summary of an issue
    - Check when an issue was created or last updated
    - Get the reporter and project information
    
    Returns comprehensive issue details including summary, description, status,
    priority, assignee, reporter, dates, and more.
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, issue_key: str) -> Dict[str, Any]:
        """Async version that handles the core logic."""
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

            # Fetch the issue
            issue = await client.get_issue(issue_key)

            return {
                "success": True,
                "issue": issue,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to fetch Jira issue {issue_key}: {str(e)}",
            }

    def run(self, issue_key: str) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(issue_key))


def get_jira_issue_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for fetching Jira issues with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for fetching Jira issues
    """
    tool_instance = GetJiraIssueTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Jira Issue",
        description="""Fetch detailed information about a Jira issue by its key (e.g., 'PROJ-123').
        
        Use this when you need to:
        - Get current status, assignee, or priority of an issue
        - Read the description and summary of an issue
        - Check when an issue was created or last updated
        - Get the reporter and project information
        
        Input:
        - issue_key (str): The Jira issue key (e.g., 'PROJ-123', 'BUG-456')
        
        Returns comprehensive issue details.""",
        args_schema=GetJiraIssueInput,
    )
