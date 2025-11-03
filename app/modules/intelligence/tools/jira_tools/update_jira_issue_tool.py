"""Tool for updating Jira issues."""

from typing import Dict, Any, Optional
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class UpdateJiraIssueInput(BaseModel):
    """Input schema for updating a Jira issue."""

    issue_key: str = Field(
        description="The Jira issue key to update (e.g., 'PROJ-123')"
    )
    summary: Optional[str] = Field(
        default=None, description="New summary/title for the issue (optional)"
    )
    description: Optional[str] = Field(
        default=None, description="New description for the issue (optional)"
    )
    priority: Optional[str] = Field(
        default=None,
        description="New priority: 'Highest', 'High', 'Medium', 'Low', 'Lowest' (optional)",
    )


class UpdateJiraIssueTool:
    """Tool for updating existing Jira issues."""

    name = "Update Jira Issue"
    description = """Update fields of an existing Jira issue.
    
    Use this tool when you need to:
    - Change the summary or description of an issue
    - Update the priority level
    - Modify issue details based on new information
    - Correct mistakes in issue fields
    
    You can update one or more fields at once. Only provide the fields you want to change.
    
    Returns the updated issue with all current details.
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(
        self,
        issue_key: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Dict[str, Any]:
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

            # Build fields dictionary
            fields = {}
            if summary is not None:
                fields["summary"] = summary
            if description is not None:
                fields["description"] = description
            if priority is not None:
                fields["priority"] = {"name": priority}

            if not fields:
                return {
                    "success": False,
                    "error": "No fields provided to update",
                    "message": "You must provide at least one field to update (summary, description, or priority)",
                }

            # Update the issue
            issue = await asyncio.to_thread(
                client.update_issue, issue_key=issue_key, fields=fields
            )

            return {
                "success": True,
                "message": f"Successfully updated issue {issue_key}",
                "updated_fields": list(fields.keys()),
                "issue": issue,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update Jira issue {issue_key}: {str(e)}",
            }

    def run(
        self,
        issue_key: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(issue_key, summary, description, priority))


def update_jira_issue_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for updating Jira issues with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for updating Jira issues
    """
    tool_instance = UpdateJiraIssueTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Update Jira Issue",
        description="""Update fields of an existing Jira issue.
        
        Use this when you need to:
        - Change the summary or description
        - Update the priority level
        - Modify issue details based on new information
        
        Inputs:
        - issue_key (str): The issue key (e.g., 'PROJ-123')
        - summary (str, optional): New summary/title
        - description (str, optional): New description
        - priority (str, optional): 'Highest', 'High', 'Medium', 'Low', 'Lowest'
        
        Provide only the fields you want to update. Returns updated issue details.""",
        args_schema=UpdateJiraIssueInput,
    )
