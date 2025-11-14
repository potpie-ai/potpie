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
        description="New priority level. Use 'Get Jira Project Details' tool to see valid priorities. (optional)",
    )
    assignee_id: Optional[str] = Field(
        default=None,
        description="Atlassian account ID to assign the issue to. Use 'Get Jira Project Users' tool to find user account IDs. (optional)",
    )


class UpdateJiraIssueTool:
    """Tool for updating existing Jira issues."""

    name = "Update Jira Issue"
    description = """Update fields of an existing Jira issue.

    TIPS:
    - Use 'Get Jira Project Details' tool to discover valid priority levels
    - Use 'Get Jira Project Users' tool to get user account IDs for assignment

    Use this tool when you need to:
    - Change the summary or description of an issue
    - Update the priority level
    - Assign or reassign the issue to a user (requires account_id)
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
        assignee_id: Optional[str] = None,
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
            if not fields and not assignee_id:
                return {
                    "success": False,
                    "error": "No fields provided to update",
                    "message": "You must provide at least one field to update (summary, description, priority, or assignee_id)",
                }

            # Update the issue
            issue = await client.update_issue(
                issue_key=issue_key,
                fields=fields,
                assignee_id=assignee_id,
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
        assignee_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(
            self.arun(issue_key, summary, description, priority, assignee_id)
        )


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

        TIP: Use 'Get Jira Project Details' to discover valid priorities.

        Use this when you need to:
        - Change the summary or description
        - Update the priority level
        - Modify issue details based on new information

        Inputs:
        - issue_key (str): The issue key (e.g., 'PROJ-123')
        - summary (str, optional): New summary/title
        - description (str, optional): New description
        - priority (str, optional): New priority level (see project details for valid values, Eg: 'High', 'Medium', 'Low')
        - assignee_id (str, optional): Atlassian account ID to assign issue to (use 'Get Jira Project Users' to find account IDs)

        Provide only the fields you want to update. Returns updated issue details.""",
        args_schema=UpdateJiraIssueInput,
    )
