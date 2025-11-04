"""Tool for linking Jira issues together."""

from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class LinkJiraIssuesInput(BaseModel):
    """Input schema for linking Jira issues."""

    issue_key: str = Field(
        description="The source issue key (e.g., 'PROJ-123')"
    )
    linked_issue_key: str = Field(
        description="The target issue key to link to (e.g., 'PROJ-456')"
    )
    link_type: str = Field(
        description="The type of link to create (e.g., 'Blocks', 'Relates', 'Duplicates', 'Clones'). Use Get Project Details tool to see available link types."
    )


class LinkJiraIssuesTool:
    """Tool for creating links between Jira issues."""

    name = "Link Jira Issues"
    description = """Create a relationship link between two Jira issues.
    
    Use this tool when you need to:
    - Link related issues together (e.g., 'Relates to')
    - Show that one issue blocks another (e.g., 'Blocks')
    - Mark duplicate issues (e.g., 'Duplicates')
    - Clone or split issues (e.g., 'Clones')
    - Create parent-child relationships for subtasks
    - Establish dependencies between issues
    
    Common link types:
    - 'Blocks' / 'is blocked by'
    - 'Relates' / 'relates to'
    - 'Duplicates' / 'is duplicated by'
    - 'Clones' / 'is cloned by'
    - 'Causes' / 'is caused by'
    
    TIP: Use the 'Get Jira Project Details' tool first to see all available
    link types for your Jira instance.
    
    Returns confirmation of the link creation.
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(
        self, issue_key: str, linked_issue_key: str, link_type: str
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

            # Create the link
            result = await asyncio.to_thread(
                client.link_issues,
                issue_key=issue_key,
                linked_issue_key=linked_issue_key,
                link_type=link_type,
            )

            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to link Jira issues {issue_key} and {linked_issue_key}: {str(e)}",
            }

    def run(
        self, issue_key: str, linked_issue_key: str, link_type: str
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(issue_key, linked_issue_key, link_type))


def link_jira_issues_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for linking Jira issues with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for linking Jira issues
    """
    tool_instance = LinkJiraIssuesTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Link Jira Issues",
        description="""Create a relationship link between two Jira issues.
        
        Use this when you need to:
        - Link related issues together
        - Show blocking relationships
        - Mark duplicates
        - Clone or split issues
        - Create dependencies
        
        Inputs:
        - issue_key (str): Source issue key (e.g., 'PROJ-123')
        - linked_issue_key (str): Target issue key (e.g., 'PROJ-456')
        - link_type (str): Type of link ('Blocks', 'Relates', 'Duplicates', 'Clones', etc.)
        
        TIP: Use 'Get Jira Project Details' tool first to see available link types.
        
        Returns confirmation of the link creation.""",
        args_schema=LinkJiraIssuesInput,
    )
