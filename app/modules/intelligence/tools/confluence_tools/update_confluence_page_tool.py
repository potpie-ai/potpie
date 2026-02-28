"""Update Confluence Page Tool

Allows agents to update existing Confluence pages.
"""

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
import asyncio
from typing import Any, Dict, Optional
from app.modules.intelligence.tools.tool_schema import OnyxTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .confluence_client import (
    get_confluence_client_for_user,
    check_confluence_integration_exists,
)


class UpdateConfluencePageInput(BaseModel):
    """Input schema for update_confluence_page tool."""

    page_id: str = Field(
        description="The ID of the page to update (get from 'Search Confluence Pages' or 'Get Confluence Page')"
    )
    version_number: int = Field(
        description="Current version number of the page. REQUIRED to prevent conflicts. Use 'Get Confluence Page' tool first to retrieve page.version.number."
    )
    title: Optional[str] = Field(
        default=None,
        description="New title for the page (optional, keeps current title if not provided)",
    )
    body: Optional[str] = Field(
        default=None,
        description="New content for the page in HTML format or plain text (optional, keeps current content if not provided)",
    )
    status: str = Field(
        default="current",
        description="Page status: 'current' (published) or 'draft'. Default: 'current'",
    )


class UpdateConfluencePageTool:
    """Tool for updating Confluence pages."""

    name = "Update Confluence Page"
    description = """Update an existing Confluence page's title or content.

    CRITICAL REQUIREMENT:
    MUST use 'Get Confluence Page' tool FIRST to retrieve the current version number.
    Confluence requires the version number to prevent edit conflicts.

    Use this tool when you need to:
    - Update documentation with new information
    - Correct errors or outdated content
    - Add sections to existing pages
    - Modify page titles
    - Update content based on code changes or feedback

    Workflow:
    1. Use 'Get Confluence Page' to get current page (returns page.version.number)
    2. Use this tool with the version number to update
    3. Version is automatically incremented after update

    Update Options:
    - Update title only: Provide title, omit body
    - Update content only: Provide body, omit title
    - Update both: Provide both title and body

    Content Format:
    - Body can be HTML or plain text
    - For HTML: Use standard tags (<p>, <h1>, <ul>, etc.)
    - For plain text: Will be wrapped in <p> tags

    After updating:
    - Page version increments automatically
    - Previous versions remain in page history
    - Use 'Add Confluence Comment' to explain changes

    Returns:
    - Updated page with new version number, title, and metadata
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
        version_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Update an existing Confluence page.

        Args:
            page_id: The page ID
            version_number: Current version number
            title: New title (optional)
            body: New content (optional)
            status: Page status

        Returns:
            Dictionary with success status and updated page data
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
                # If neither title nor body provided, return error
                if title is None and body is None:
                    return {
                        "success": False,
                        "error": "Must provide at least title or body to update",
                    }

                # Get current page to use existing values if not provided
                current_page = await client.get_page(page_id=page_id)

                # Use provided values or fall back to current
                update_title = title if title is not None else current_page.get("title")
                update_body = body
                if update_body is None:
                    # Extract current body
                    current_body = current_page.get("body", {})
                    if "storage" in current_body:
                        update_body = current_body["storage"].get("value", "")
                    else:
                        update_body = ""

                # Update page
                page = await client.update_page(
                    page_id=page_id,
                    version_number=version_number,
                    title=update_title,
                    body=update_body,
                    status=status,
                )

                # Extract updated page information
                page_data = {
                    "id": page.get("id"),
                    "status": page.get("status"),
                    "title": page.get("title"),
                    "space_id": page.get("spaceId"),
                    "parent_id": page.get("parentId"),
                    "version": page.get("version", {}).get("number"),
                    "author_id": page.get("authorId"),
                    "created_at": page.get("createdAt"),
                    "_links": page.get("_links", {}),
                }

                return {
                    "success": True,
                    "page": page_data,
                    "message": f"Successfully updated page {page_id} to version {page.get('version', {}).get('number')}",
                }
            finally:
                await client.close()

        except Exception as e:
            logger.error(f"Error updating Confluence page: {str(e)}")
            return {"success": False, "error": str(e)}

    def run(
        self,
        page_id: str,
        version_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        status: str = "current",
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(page_id, version_number, title, body, status))


def update_confluence_page_tool(db: Session, user_id: str) -> OnyxTool:
    """
    Create a tool for updating Confluence pages with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Confluence integration

    Returns:
        A configured OnyxTool for updating Confluence pages
    """
    tool_instance = UpdateConfluencePageTool(db, user_id)

    return OnyxTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Update Confluence Page",
        description="""Update an existing Confluence page's title or content.

        CRITICAL: MUST use 'Get Confluence Page' tool FIRST to get current version number.

        Use this when you need to:
        - Update documentation with new information
        - Correct errors or outdated content
        - Modify page titles or add new sections

        Inputs:
        - page_id (str): Page ID to update
        - version_number (int): Current version number (REQUIRED - get from 'Get Confluence Page')
        - title (str, optional): New title (omit to keep current)
        - body (str, optional): New content in HTML or plain text (omit to keep current)
        - status (str): 'current' or 'draft' (default: 'current')

        Required Workflow:
        1. Call 'Get Confluence Page' to get page.version.number
        2. Call this tool with that version number
        3. Version increments automatically

        Must provide at least title OR body to update.

        Content Tips:
        - Use HTML for formatting: <h1>, <p>, <ul>, <strong>, etc.
        - Plain text is wrapped in <p> tags

        Returns updated page with new version number and metadata.""",
        args_schema=UpdateConfluencePageInput,
    )
