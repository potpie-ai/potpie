"""Tool for adding comments to Jira issues."""

from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class AddJiraCommentInput(BaseModel):
    """Input schema for adding a comment to a Jira issue."""

    issue_key: str = Field(
        description="The Jira issue key to comment on (e.g., 'PROJ-123')"
    )
    comment: str = Field(description="The comment text to add to the issue")


class AddJiraCommentTool:
    """Tool for adding comments to Jira issues."""

    name = "Add Jira Comment"
    description = """Add a comment to an existing Jira issue.
    
    Use this tool when you need to:
    - Add analysis results or findings to an issue
    - Document progress or updates on work
    - Provide additional context or information
    - Log investigation results or debugging notes
    - Reply to discussions on an issue
    - Add code review feedback
    
    The comment will be posted with your user account as the author.
    Supports Jira text formatting (markdown-like syntax).
    
    Returns confirmation and the posted comment details.
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, issue_key: str, comment: str) -> Dict[str, Any]:
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

            # Add the comment
            result = await client.add_comment(
                issue_key=issue_key, comment_body=comment
            )

            return {
                "success": True,
                "message": f"Successfully added comment to issue {issue_key}",
                "comment": result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to add comment to Jira issue {issue_key}: {str(e)}",
            }

    def run(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(issue_key, comment))


def add_jira_comment_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for adding comments to Jira issues with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for adding Jira comments
    """
    tool_instance = AddJiraCommentTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Add Jira Comment",
        description="""Add a comment to an existing Jira issue.
        
        Use this when you need to:
        - Document analysis results or findings
        - Provide updates or progress on work
        - Add additional context or information
        - Log investigation or debugging notes
        - Reply to discussions
        
        Inputs:
        - issue_key (str): The issue key (e.g., 'PROJ-123')
        - comment (str): The comment text to add
        
        Returns confirmation and posted comment details.""",
        args_schema=AddJiraCommentInput,
    )
