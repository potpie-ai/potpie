"""Create Confluence Page Tool

Allows agents to create new Confluence pages in a space.
"""

import logging
import asyncio
from typing import Any, Dict, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .confluence_client import (
    get_confluence_client_for_user,
    check_confluence_integration_exists,
)


class CreateConfluencePageInput(BaseModel):
    """Input schema for create_confluence_page tool."""

    space_id: str = Field(
        description="The numeric space ID from 'Get Confluence Spaces' tool (the 'id' field, NOT the 'key' field). Example: '1245186'"
    )
    title: str = Field(description="The title of the new page (required)")
    body: str = Field(
        description="The content of the page in HTML format or plain text (required)"
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="Optional parent page ID to create this as a child page. Use 'Search Confluence Pages' or 'Get Confluence Space Pages' to find parent page IDs.",
    )
    status: str = Field(
        default="current",
        description="Page status: 'current' (published immediately) or 'draft' (saved as draft). Default: 'current'",
    )


class CreateConfluencePageTool:
    """Tool for creating Confluence pages."""

    name = "Create Confluence Page"
    description = """Create a new documentation page in Confluence.
    
    Use this tool when you need to:
    - Create new documentation pages
    - Document a new feature or API
    - Create meeting notes or project documentation
    - Add a new page to existing documentation structure
    - Create a child page under an existing parent
    
    IMPORTANT Prerequisites:
    1. Get numeric space_id first using 'Get Confluence Spaces' tool (use the 'id' field, NOT the 'key')
    2. If creating a child page, get parent_id using 'Search Confluence Pages' or 'Get Confluence Space Pages'
    
    Content Format:
    - Body can be HTML or plain text
    - For HTML: Use standard HTML tags (<p>, <h1>, <ul>, etc.)
    - For plain text: Will be wrapped in <p> tags automatically
    
    Page Hierarchy:
    - Omit parent_id to create a top-level page in the space
    - Provide parent_id to create a child page (nested documentation)
    
    Status Options:
    - 'current': Page is published immediately (visible to all users)
    - 'draft': Page is saved as draft (visible only to you)
    
    After creating:
    - Use 'Update Confluence Page' to modify content later
    - Use 'Add Confluence Comment' to add comments
    
    Returns:
    - Created page with ID, version, status, and view URL
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
        space_id: str,
        title: str,
        body: str,
        parent_id: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Create a new Confluence page.

        Args:
            space_id: The space ID
            title: Page title
            body: Page content (HTML or plain text)
            parent_id: Optional parent page ID
            status: Page status (current or draft)

        Returns:
            Dictionary with success status and created page data
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

            # Create page
            page = client.create_page(
                space_id=space_id,
                title=title,
                body=body,
                parent_id=parent_id,
                status=status,
            )

            # Extract created page information
            page_data = {
                "id": page.get("id"),
                "status": page.get("status"),
                "title": page.get("title"),
                "space_id": page.get("spaceId"),
                "parent_id": page.get("parentId"),
                "parent_type": page.get("parentType"),
                "author_id": page.get("authorId"),
                "created_at": page.get("createdAt"),
                "version": page.get("version", {}).get("number"),
                "_links": page.get("_links", {}),
            }

            client.close()

            return {
                "success": True,
                "page": page_data,
                "message": f"Successfully created page '{title}' in space {space_id}",
            }

        except Exception as e:
            logging.error(f"Error creating Confluence page: {str(e)}")
            return {"success": False, "error": str(e)}

    def run(
        self,
        space_id: str,
        title: str,
        body: str,
        parent_id: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(space_id, title, body, parent_id, status))


def create_confluence_page_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for creating Confluence pages with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Confluence integration

    Returns:
        A configured StructuredTool for creating Confluence pages
    """
    tool_instance = CreateConfluencePageTool(db, user_id)

    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Create Confluence Page",
        description="""Create a new documentation page in a Confluence space.
        
        Use this when you need to:
        - Create new documentation pages
        - Document features, APIs, or processes
        - Add pages to existing documentation structure
        - Create child pages under existing pages
        
        Inputs:
        - space_id (str): Numeric space ID (get 'id' field from 'Get Confluence Spaces', NOT 'key'). Example: '1245186'
        - title (str): Page title (required)
        - body (str): Page content in HTML or plain text (required)
        - parent_id (str, optional): Parent page ID to create as child page (get from 'Search Confluence Pages')
        - status (str): 'current' (published) or 'draft' (default: 'current')
        
        CRITICAL: Use the numeric space 'id' field from 'Get Confluence Spaces', NOT the 'key' field.
        The 'id' is numeric like '1245186', while 'key' is alphanumeric like 'OOP' or '~712020...'.
        
        Content Tips:
        - Use HTML for rich formatting: <h1>, <p>, <ul>, <li>, <strong>, etc.
        - Plain text is automatically wrapped in <p> tags
        
        After creating:
        - Use 'Update Confluence Page' to modify (requires version number from this response)
        - Use 'Add Confluence Comment' to add comments
        
        Returns created page with ID, version, status, and view URL.""",
        args_schema=CreateConfluencePageInput,
    )
