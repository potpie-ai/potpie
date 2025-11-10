"""Tool for fetching all Jira projects accessible to the user."""

from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class GetJiraProjectsInput(BaseModel):
    """Input schema for getting Jira projects."""

    max_results: int = Field(
        default=50,
        description="Maximum number of projects to return (default: 50, max: 100)",
    )


class GetJiraProjectsTool:
    """Tool for fetching all Jira projects accessible to the user."""

    name = "Get Jira Projects"
    description = """Fetch all Jira projects accessible to the authenticated user.
    
    Use this tool when you need to:
    - List all available projects for the user
    - Find a project by name to get its key
    - Show the user what projects they can create issues in
    - Get project metadata (ID, key, name, lead, type)
    
    Returns a list of all accessible projects with their details including:
    - Project key (used for creating issues)
    - Project name
    - Project ID
    - Project type (software, business, service_desk)
    - Project lead
    - Project URL
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, max_results: int = 50) -> Dict[str, Any]:
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

            # Fetch projects
            # Limit max_results to reasonable bounds
            max_results = min(max(1, max_results), 100)

            result = await client.get_projects(
                start_at=0, max_results=max_results
            )

            return {
                "success": True,
                "total": result["total"],
                "returned": len(result["projects"]),
                "projects": result["projects"],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to fetch Jira projects: {str(e)}",
            }

    def run(self, max_results: int = 50) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(max_results))


def get_jira_projects_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for fetching Jira projects with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for fetching Jira projects
    """
    tool_instance = GetJiraProjectsTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Jira Projects",
        description="""Fetch all Jira projects accessible to the authenticated user.
        
        Use this when you need to:
        - List all available projects
        - Find a project key by name
        - Show what projects the user can work with
        - Get project metadata for creating issues
        
        Input:
        - max_results (int, optional): Maximum projects to return (default: 50, max: 100)
        
        Returns list of projects with key, name, ID, type, lead, and URL.""",
        args_schema=GetJiraProjectsInput,
    )
