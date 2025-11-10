"""Tool for transitioning Jira issues to different statuses."""

from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class TransitionJiraIssueInput(BaseModel):
    """Input schema for transitioning a Jira issue."""

    issue_key: str = Field(
        description="The Jira issue key to transition (e.g., 'PROJ-123')"
    )
    transition: str = Field(
        description="The name of the transition/status to move to (e.g., 'In Progress', 'Done', 'To Do')"
    )


class TransitionJiraIssueTool:
    """Tool for transitioning Jira issues to different statuses."""

    name = "Transition Jira Issue"
    description = """Change the status of a Jira issue by transitioning it through the workflow.
    
    Use this tool when you need to:
    - Move an issue to 'In Progress' when work starts
    - Mark an issue as 'Done' when completed
    - Move an issue back to 'To Do' if work needs to restart
    - Close or resolve an issue
    - Move through any custom workflow states
    - Change issue status based on external events or conditions
    
    The transition name should match your Jira workflow (case-insensitive).
    Common transitions: 'To Do', 'In Progress', 'Done', 'Closed', 'Resolved'
    
    If the transition name doesn't exist, the error will list available transitions.
    
    Returns the updated issue with its new status.
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, issue_key: str, transition: str) -> Dict[str, Any]:
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

            # Transition the issue
            issue = await client.transition_issue(
                issue_key=issue_key, transition_name=transition
            )

            return {
                "success": True,
                "message": f"Successfully transitioned issue {issue_key} to '{transition}'",
                "new_status": issue.get("status"),
                "issue": issue,
            }
        except Exception as e:
            error_msg = str(e)

            # If transition not found, try to get available transitions
            if (
                "not found" in error_msg.lower()
                or "available transitions" in error_msg.lower()
            ):
                try:
                    client = await get_jira_client_for_user(self.user_id, self.db)
                    transitions = await client.get_transitions(
                        issue_key=issue_key
                    )
                    available = [t["name"] for t in transitions]
                    error_msg = f"{error_msg}\n\nAvailable transitions for {issue_key}: {', '.join(available)}"
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "message": f"Failed to transition Jira issue {issue_key}: {str(e)}",
                    }

            return {
                "success": False,
                "error": error_msg,
                "message": f"Failed to transition Jira issue {issue_key}: {error_msg}",
            }

    def run(self, issue_key: str, transition: str) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(issue_key, transition))


def transition_jira_issue_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for transitioning Jira issues with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for transitioning Jira issues
    """
    tool_instance = TransitionJiraIssueTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Transition Jira Issue",
        description="""Change the status of a Jira issue through its workflow.
        
        Use this when you need to:
        - Start work on an issue (move to 'In Progress')
        - Complete work (move to 'Done' or 'Resolved')
        - Reopen issues or move back to earlier states
        - Move through custom workflow states
        
        Inputs:
        - issue_key (str): The issue key (e.g., 'PROJ-123')
        - transition (str): Target status name ('To Do', 'In Progress', 'Done', etc.)
        
        If transition name is invalid, error will list available transitions.
        Returns updated issue with new status.""",
        args_schema=TransitionJiraIssueInput,
    )
