"""Tool for getting comprehensive Jira project details."""

from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class GetJiraProjectDetailsInput(BaseModel):
    """Input schema for getting Jira project details."""

    project_key: str = Field(
        description="The project key (e.g., 'PROJ', 'BUG') to get details for"
    )


class GetJiraProjectDetailsTool:
    """Tool for fetching comprehensive Jira project details."""

    name = "Get Jira Project Details"
    description = """Get comprehensive details about a Jira project including all metadata.

    Use this tool when you need to:
    - Find out what issue types are available in a project (Task, Bug, Story, Epic, etc.)
    - Get available priority levels (Highest, High, Medium, Low, Lowest)
    - See all possible statuses/workflows for transitions (To Do, In Progress, Done, etc.)
    - Get available issue link types (Blocks, Relates to, Duplicates, etc.)
    - View existing labels in the project
    - Get project lead and basic information

    This tool is essential BEFORE creating or updating issues to ensure you use valid
    issue types and priorities. The statuses are for use with the Transition tool.

    Returns comprehensive project metadata including:
    - Project basic info (name, description, lead, URL)
    - All available issue types with descriptions
    - All priority levels
    - All statuses and their categories
    - All issue link types
    - Existing project labels
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, project_key: str) -> Dict[str, Any]:
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

            # Fetch project details
            details = await asyncio.to_thread(
                client.get_project_details, project_key=project_key
            )

            return {
                "success": True,
                "project": details,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to fetch Jira project details for {project_key}: {str(e)}",
            }

    def run(self, project_key: str) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(project_key))


def get_jira_project_details_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for fetching comprehensive Jira project details with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for fetching Jira project details
    """
    tool_instance = GetJiraProjectDetailsTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Jira Project Details",
        description="""Get comprehensive metadata about a Jira project.

        IMPORTANT: Use this tool FIRST before creating or updating issues to discover:
        - Valid issue types for creation (Task, Bug, Story, Epic, etc.)
        - Valid priority levels for creation/updates (Highest, High, Medium, Low, Lowest)
        - Valid statuses for transitions (To Do, In Progress, Done, etc.) - use with Transition tool
        - Available link types for linking issues
        - Existing labels in the project

        Input:
        - project_key (str): The project key (e.g., 'PROJ', 'BUG')

        Returns all project metadata including issue types, priorities, statuses,
        link types, labels, and basic project information.""",
        args_schema=GetJiraProjectDetailsInput,
    )
