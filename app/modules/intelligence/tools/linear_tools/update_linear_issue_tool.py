from typing import Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import StructuredTool

from app.modules.intelligence.tools.linear_tools.linear_client import get_linear_client

class UpdateLinearIssueInput(BaseModel):
    issue_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    assignee_id: Optional[str] = None
    priority: Optional[int] = None
    comment: Optional[str] = None

def update_linear_issue(
    issue_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    assignee_id: Optional[str] = None,
    priority: Optional[int] = None,
    comment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update a Linear issue with the provided details.
    
    Args:
        issue_id: The ID of the Linear issue to update
        title: New title for the issue (optional)
        description: New description for the issue (optional)
        status: New status for the issue (optional)
        assignee_id: ID of the user to assign the issue to (optional)
        priority: New priority for the issue (optional)
        comment: Comment to add to the issue (optional)
        
    Returns:
        Dict containing the updated issue details
    """
    try:
        client = get_linear_client()
        
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
            "assignee": issue["assignee"]["name"] if issue.get("assignee") else None,
            "priority": issue["priority"],
            "updated_at": str(issue["updatedAt"]),
            "comment_added": bool(comment_result and comment_result["success"] if comment_result else False)
        }
    except Exception as e:
        raise ValueError(f"Error updating Linear issue: {str(e)}")

def update_linear_issue_tool() -> StructuredTool:
    """Create a tool for updating Linear issues"""
    return StructuredTool.from_function(
        func=update_linear_issue,
        name="update_linear_issue",
        description="Update a Linear issue with new details and optionally add a comment",
        args_schema=UpdateLinearIssueInput
    ) 