"""Add Confluence Comment Tool

Allows agents to add comments to Confluence pages.
"""

import asyncio
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .confluence_client import (
    get_confluence_client_for_user,
    check_confluence_integration_exists,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class AddConfluenceCommentInput(BaseModel):
    """Input schema for add_confluence_comment tool."""

    page_id: str = Field(
        description="The ID of the page to comment on (get from 'Search Confluence Pages' or 'Get Confluence Page')"
    )
    comment: str = Field(description="The comment text (plain text or HTML format)")
    parent_comment_id: Optional[str] = Field(
        default=None,
        description="Optional ID of parent comment to reply to. Omit to create a top-level comment. Provide to create a reply thread.",
    )
    status: str = Field(
        default="current",
        description="Comment status: 'current' (published immediately) or 'draft' (saved as draft). Default: 'current'",
    )


class AddConfluenceCommentTool:
    """Tool for adding comments to Confluence pages."""

    name = "Add Confluence Comment"
    description = """Add a comment to a Confluence page or reply to an existing comment.

    Use this tool when you need to:
    - Provide feedback on documentation
    - Ask questions about page content
    - Suggest improvements or corrections
    - Reply to existing comments or discussions
    - Explain changes made to a page
    - Add notes without modifying the main page content

    Comment Types:
    - Top-level comment: Omit parent_comment_id (comments on the page itself)
    - Reply comment: Provide parent_comment_id (creates a threaded discussion)

    Content Format:
    - Comment can be plain text or HTML
    - For HTML: Use standard tags (<p>, <strong>, <em>, <ul>, etc.)
    - For plain text: Will be wrapped in <p> tags

    Best Practices:
    - Use comments for questions, feedback, or discussion
    - Use 'Update Confluence Page' to modify actual page content
    - Comment on pages after making major updates to explain changes
    - Reply to existing comments to continue discussions

    After adding comment:
    - Comment appears on the page for all users to see
    - Users receive notifications about new comments
    - Comments can be edited or deleted later via Confluence UI

    Returns:
    - Created comment with ID, author, timestamp, and view URL
    """

    def __init__(self, db: Session, user_id: str):
        """
        Initialize the tool.

        Args:
            db: Database session
            user_id: The user ID
        """
        self.db = db
        self.user_id = user_id

    async def arun(
        self,
        page_id: str,
        comment: str,
        parent_comment_id: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Add a comment to a Confluence page.

        Args:
            page_id: The page ID
            comment: Comment text
            parent_comment_id: Optional parent comment for replies
            status: Comment status

        Returns:
            Dictionary with success status and comment data
        """
        try:
            # Check if integration exists
            has_integration = await check_confluence_integration_exists(
                self.user_id, self.db
            )
            if not has_integration:
                return {
                    "success": False,
                    "error": "No Confluence integration found. Please connect your Confluence account first.",
                }

            # Get Confluence client
            client = await get_confluence_client_for_user(self.user_id, self.db)
            if not client:
                return {
                    "success": False,
                    "error": "Failed to initialize Confluence client",
                }

            try:
                # Add comment
                comment_response = await client.add_comment(
                    page_id=page_id,
                    comment=comment,
                    parent_comment_id=parent_comment_id,
                    status=status,
                )

                # Extract comment information
                comment_data = {
                    "id": comment_response.get("id"),
                    "status": comment_response.get("status"),
                    "title": comment_response.get("title"),
                    "page_id": comment_response.get("pageId"),
                    "parent_comment_id": comment_response.get("parentCommentId"),
                    "version": comment_response.get("version", {}).get("number"),
                    "author_id": comment_response.get("authorId"),
                    "created_at": comment_response.get("createdAt"),
                    "_links": comment_response.get("_links", {}),
                }

                comment_type = "reply" if parent_comment_id else "comment"
                return {
                    "success": True,
                    "comment": comment_data,
                    "message": f"Successfully added {comment_type} to page {page_id}",
                }
            finally:
                await client.close()

        except Exception as e:
            logger.error(f"Error adding Confluence comment: {str(e)}")
            return {"success": False, "error": str(e)}

    def run(
        self,
        page_id: str,
        comment: str,
        parent_comment_id: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(page_id, comment, parent_comment_id, status))


def add_confluence_comment_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for adding comments to Confluence pages with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Confluence integration

    Returns:
        A configured StructuredTool for adding Confluence comments
    """
    tool_instance = AddConfluenceCommentTool(db, user_id)

    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Add Confluence Comment",
        description="""Add a comment to a Confluence page or reply to an existing comment.

        Use this when you need to:
        - Provide feedback on documentation
        - Ask questions about page content
        - Reply to existing comments (create discussion threads)
        - Explain changes made to a page
        - Add notes without modifying page content

        Inputs:
        - page_id (str): Page ID to comment on (get from 'Search Confluence Pages' or 'Get Confluence Page')
        - comment (str): Comment text (plain text or HTML)
        - parent_comment_id (str, optional): Parent comment ID to create a reply (omit for top-level comment)
        - status (str): 'current' (published) or 'draft' (default: 'current')

        Comment Types:
        - Top-level: Omit parent_comment_id (comments on page)
        - Reply: Provide parent_comment_id (threaded discussion)

        Content Tips:
        - Use plain text for simple comments
        - Use HTML for formatted comments: <strong>, <em>, <ul>, etc.

        Best Practices:
        - Use comments for discussion, not content changes
        - Use 'Update Confluence Page' to modify page content
        - Comment after major updates to explain changes

        Returns created comment with ID, author, timestamp, and URL.""",
        args_schema=AddConfluenceCommentInput,
    )
