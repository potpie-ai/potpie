from typing import Dict, Any, Optional
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.linear_tools.linear_client import (
    get_linear_client_for_user,
)
from app.modules.key_management.secret_manager import SecretStorageHandler


class UpdateLinearIssueInput(BaseModel):
    issue_id: str = Field(description="The ID of the Linear issue to update")
    title: Optional[str] = Field(None, description="New title for the issue")
    description: Optional[str] = Field(
        None, description="New description for the issue"
    )
    status: Optional[str] = Field(None, description="New status for the issue")
    assignee_id: Optional[str] = Field(
        None, description="ID of the user to assign the issue to"
    )
    priority: Optional[int] = Field(None, description="New priority for the issue")
    comment: Optional[str] = Field(None, description="Comment to add to the issue")


class UpdateLinearIssueTool:
    name = "Update Linear Issue"
    description = """Update a Linear issue with new details and optionally add a comment.
        :param issue_id: string, the ID of the Linear issue to update
        :param title: string, optional, new title for the issue
        :param description: string, optional, new description for the issue
        :param status: string, optional, new status for the issue
        :param assignee_id: string, optional, ID of the user to assign the issue to
        :param priority: integer, optional, new priority for the issue
        :param comment: string, optional, comment to add to the issue

        Returns dictionary containing the updated issue details.
        """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(
        self,
        issue_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        assignee_id: Optional[str] = None,
        priority: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version that handles the core logic"""
        try:
            # Check if user has Linear API key configured
            has_key = await SecretStorageHandler.check_secret_exists(
                service="linear",
                customer_id=self.user_id,
                service_type="integration",
                db=self.db,
            )

            if not has_key:
                return {
                    "error": "Please head to the Key Management screen and add your Linear API Key in order to use Linear operations"
                }

            # Get the user-specific client
            client = await get_linear_client_for_user(self.user_id, self.db)

            # Prepare update data
            update_data = {}
            if title is not None:
                update_data["title"] = title
            if description is not None:
                update_data["description"] = description
            if status is not None:
                update_data["stateId"] = status
            if assignee_id is not None:
                update_data["assigneeId"] = assignee_id
            if priority is not None:
                update_data["priority"] = priority

            # Update the issue if there are changes
            if update_data:
                result = client.update_issue(issue_id, update_data)
                issue = result["issue"]
            else:
                # If no updates, fetch current issue state
                issue = client.get_issue(issue_id)

            # Add comment if provided
            comment_result = None
            if comment:
                comment_result = client.comment_create(issue_id, comment)

            # Return updated issue details
            return {
                "id": issue["id"],
                "title": issue["title"],
                "description": issue["description"],
                "status": issue["state"]["name"] if issue.get("state") else None,
                "assignee": (
                    issue["assignee"]["name"] if issue.get("assignee") else None
                ),
                "priority": issue["priority"],
                "updated_at": str(issue["updatedAt"]),
                "comment_added": bool(
                    comment_result and comment_result["success"]
                    if comment_result
                    else False
                ),
            }
        except Exception as e:
            raise ValueError(f"Error updating Linear issue: {str(e)}")

    def run(
        self,
        issue_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        assignee_id: Optional[str] = None,
        priority: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version"""
        return asyncio.run(
            self.arun(
                issue_id, title, description, status, assignee_id, priority, comment
            )
        )


def update_linear_issue_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for updating Linear issues with user context.

    Args:
        db: Database session for secret retrieval
        user_id: The user ID to fetch their specific Linear API key

    Returns:
        A configured StructuredTool for updating Linear issues
    """
    tool_instance = UpdateLinearIssueTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Update Linear Issue",
        description="""Update a Linear issue with new details and optionally add a comment.
                       Inputs for the run method:
                       - issue_id (str): The ID of the Linear issue to update
                       - title (str, optional): New title for the issue
                       - description (str, optional): New description for the issue
                       - status (str, optional): New status for the issue
                       - assignee_id (str, optional): ID of the user to assign the issue to
                       - priority (int, optional): New priority for the issue
                       - comment (str, optional): Comment to add to the issue""",
        args_schema=UpdateLinearIssueInput,
    )
