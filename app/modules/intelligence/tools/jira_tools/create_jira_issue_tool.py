"""Tool for creating new Jira issues."""

from typing import Dict, Any, Optional, List
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session
import logging

from app.modules.intelligence.tools.jira_tools.jira_client import (
    get_jira_client_for_user,
    check_jira_integration_exists,
)


class CreateJiraIssueInput(BaseModel):
    """Input schema for creating a Jira issue."""

    project_key: str = Field(
        description="The project key where the issue will be created (e.g., 'PROJ', 'BUG')"
    )
    summary: str = Field(description="Brief summary/title of the issue (required)")
    description: str = Field(description="Detailed description of the issue")
    issue_type: str = Field(
        default="To Do",
        description="Type of issue: 'To Do', 'In Progress', 'Completed'. (default: 'To Do')",
    )
    priority: Optional[str] = Field(
        default=None,
        description="Priority: 'Highest', 'High', 'Medium', 'Low', 'Lowest' (optional)",
    )
    labels: Optional[List[str]] = Field(
        default=None, description="List of labels to add to the issue (optional)"
    )


class CreateJiraIssueTool:
    """Tool for creating new Jira issues."""

    name = "Create Jira Issue"
    description = """Create a new issue in Jira.
    
    Use this tool when you need to:
    - Create a bug report from error logs or user reports
    - Create a task for work that needs to be done
    - Create a story for a new feature
    - Document technical debt or improvements
    - Log incidents or problems
    
    Required fields:
    - project_key: The project where issue will be created
    - summary: Brief title/summary of the issue
    - description: Detailed description
    
    Optional fields:
    - issue_type: To Do, In Progress, Done (default: To Do)
    - priority: Highest, High, Medium, Low, Lowest
    - labels: Tags for categorization
    
    Returns the created issue with its key and details.
    """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(
        self,
        project_key: str,
        summary: str,
        description: str,
        issue_type: str = "To Do",
        priority: Optional[str] = None,
        labels: Optional[List[str]] = None,
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

            # Prepare kwargs for optional fields
            kwargs = {}
            if priority:
                kwargs["priority"] = priority
            if labels:
                kwargs["labels"] = labels

            # Create the issue
            issue = await asyncio.to_thread(
                client.create_issue,
                project_key=project_key,
                summary=summary,
                description=description,
                issue_type=issue_type,
                **kwargs,
            )

            return {
                "success": True,
                "message": f"Successfully created issue {issue['key']}",
                "issue": issue,
            }
        except Exception as e:
            logging.exception("Error creating Jira issue: %s", str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create Jira issue: {str(e)}",
            }

    def run(
        self,
        project_key: str,
        summary: str,
        description: str,
        issue_type: str = "Task",
        priority: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(
            self.arun(project_key, summary, description, issue_type, priority, labels)
        )


def create_jira_issue_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for creating Jira issues with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Jira integration

    Returns:
        A configured StructuredTool for creating Jira issues
    """
    tool_instance = CreateJiraIssueTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Create Jira Issue",
        description="""Create a new issue in Jira with summary, description, and optional fields.
        
        Use this when you need to:
        - Create bug reports from errors or issues found
        - Create tasks for work that needs to be done
        - Document feature requests or improvements
        - Log incidents or problems
        
        Inputs:
        - project_key (str): Project key (e.g., 'PROJ', 'BUG')
        - summary (str): Brief title of the issue
        - description (str): Detailed description
        - issue_type (str): 'Task', 'Bug', 'Story', 'Epic' (default: 'Task')
        - priority (str, optional): 'Highest', 'High', 'Medium', 'Low', 'Lowest'
        - labels (list, optional): List of labels/tags
        
        Returns the created issue with its key and URL.""",
        args_schema=CreateJiraIssueInput,
    )
