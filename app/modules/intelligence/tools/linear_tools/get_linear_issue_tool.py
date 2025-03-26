from typing import Dict, Any
from pydantic import BaseModel
from langchain_core.tools import StructuredTool

from app.modules.intelligence.tools.linear_tools.linear_client import get_linear_client

class GetLinearIssueInput(BaseModel):
    issue_id: str

def get_linear_issue(issue_id: str) -> Dict[str, Any]:
    """
    Fetch details of a Linear issue by its ID.
    
    Args:
        issue_id: The ID of the Linear issue to fetch
        
    Returns:
        Dict containing issue details including title, description, status, etc.
    """
    try:
        client = get_linear_client()
        issue = client.get_issue(issue_id)
        
        return {
            "id": issue["id"],
            "title": issue["title"],
            "description": issue["description"],
            "status": issue["state"]["name"] if issue.get("state") else None,
            "assignee": issue["assignee"]["name"] if issue.get("assignee") else None,
            "team": issue["team"]["name"] if issue.get("team") else None,
            "priority": issue["priority"],
            "url": issue["url"],
            "created_at": str(issue["createdAt"]),
            "updated_at": str(issue["updatedAt"])
        }
    except Exception as e:
        raise ValueError(f"Error fetching Linear issue: {str(e)}")

def get_linear_issue_tool() -> StructuredTool:
    """Create a tool for fetching Linear issues"""
    return StructuredTool.from_function(
        func=get_linear_issue,
        name="get_linear_issue",
        description="Fetch details of a Linear issue by its ID",
        args_schema=GetLinearIssueInput
    ) 