"""Tool for searching Jira issues using JQL."""

from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class SearchJiraIssuesInput(BaseModel):
    """Input schema for searching Jira issues."""

    jql: str = Field(
        description="JQL (Jira Query Language) query string. Examples: 'project = PROJ AND status = Open', 'assignee = currentUser() AND priority = High', 'labels = bug AND created >= -7d'"
    )
    max_results: int = Field(
        default=50, description="Maximum number of results to return (default: 50)"
    )
    include_comments: bool = Field(
        default=False,
        description="If True, include all comments for each issue in the results. Set to False for faster searches when comments aren't needed. (default: False)",
    )


class SearchJiraIssuesTool:
    """Tool for searching Jira issues using JQL."""

    name = "Search Jira Issues"
    description = """Search for Jira issues using JQL (Jira Query Language).

    Use this tool when you need to:
    - Find all issues in a project
    - Search for issues by status, priority, assignee, etc.
    - Find issues created/updated within a time range
    - Search by labels, components, or custom fields
    - Get unassigned issues
    - Find issues assigned to current user

    JQL Examples:
    - "project = PROJ AND status = Open"
    - "assignee = currentUser() AND priority = High"
    - "labels = bug AND created >= -7d"
    - "status changed to Done during (-1w, now())"
    - "project = PROJ ORDER BY created DESC"

    Optional parameters:
    - max_results: Limit number of results (default: 50)
    - include_comments: Set to True to include all comments for each issue (slower)

    Returns a list of matching issues with their details.
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(
        self, jql: str, max_results: int = 50, include_comments: bool = False
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

            # Search for issues
            results = await client.search_issues(
                jql=jql,
                max_results=max_results,
                include_comments=include_comments,
            )

            return {
                "success": True,
                "jql": jql,
                "returned": len(results["issues"]),
                "max_results": max_results,
                "is_last": results.get("is_last", True),
                "issues": results["issues"],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to search Jira issues: {str(e)}",
            }

    def run(
        self, jql: str, max_results: int = 50, include_comments: bool = False
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(jql, max_results, include_comments))


def search_jira_issues_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for searching Jira issues with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for searching Jira issues
    """
    tool_instance = SearchJiraIssuesTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Search Jira Issues",
        description="""Search for Jira issues using JQL (Jira Query Language).

        Use this when you need to find issues by:
        - Project, status, priority, assignee
        - Labels, components, or custom fields
        - Date ranges (created/updated)
        - Complex queries combining multiple criteria

        Inputs:
        - jql (str): JQL query string (e.g., "project = PROJ AND status = Open")
        - max_results (int): Maximum number of results (default: 50)
        - include_comments (bool): If True, include all comments for each issue (default: False)

        Note: Setting include_comments=True may slow down searches with many results.

        Returns list of matching issues with details (and comments if requested).""",
        args_schema=SearchJiraIssuesInput,
    )
