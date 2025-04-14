from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.linear_tools.linear_client import (
    get_linear_client_for_user,
)
from app.modules.key_management.secret_manager import SecretStorageHandler


class GetLinearIssueInput(BaseModel):
    issue_id: str = Field(description="The ID of the Linear issue to fetch")


class GetLinearIssueTool:
    name = "Get Linear Issue"
    description = """Fetch details of a Linear issue by its ID.
        :param issue_id: string, the ID of the Linear issue to fetch.

        Returns dictionary containing issue details including title, description, status, etc.
        """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, issue_id: str) -> Dict[str, Any]:
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

            # Fetch the issue
            issue = client.get_issue(issue_id)

            return {
                "id": issue["id"],
                "title": issue["title"],
                "description": issue["description"],
                "status": issue["state"]["name"] if issue.get("state") else None,
                "assignee": (
                    issue["assignee"]["name"] if issue.get("assignee") else None
                ),
                "team": issue["team"]["name"] if issue.get("team") else None,
                "priority": issue["priority"],
                "url": issue["url"],
                "created_at": str(issue["createdAt"]),
                "updated_at": str(issue["updatedAt"]),
            }
        except Exception as e:
            return {"error": f"Error fetching Linear issue: {str(e)}"}

    def run(self, issue_id: str) -> Dict[str, Any]:
        """Synchronous version that runs the async version"""
        return asyncio.run(self.arun(issue_id))


def get_linear_issue_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for fetching Linear issues with user context.

    Args:
        db: Database session for secret retrieval
        user_id: The user ID to fetch their specific Linear API key

    Returns:
        A configured StructuredTool for fetching Linear issues
    """
    tool_instance = GetLinearIssueTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Linear Issue",
        description="""Fetch details of a Linear issue by its ID.
                       Inputs for the run method:
                       - issue_id (str): The ID of the Linear issue to fetch.""",
        args_schema=GetLinearIssueInput,
    )
