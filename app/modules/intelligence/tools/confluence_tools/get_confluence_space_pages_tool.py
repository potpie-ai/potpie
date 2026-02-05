"""Get Confluence Space Pages Tool

Allows agents to list all pages within a specific Confluence space.
"""

import asyncio
from typing import Any, Dict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .confluence_client import (
    get_confluence_client_for_user,
    check_confluence_integration_exists,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class GetConfluenceSpacePagesInput(BaseModel):
    """Input schema for get_confluence_space_pages tool."""

    space_id: str = Field(
        description="The numeric space ID from 'Get Confluence Spaces' tool (the 'id' field, NOT the 'key' field). Example: '1245186'"
    )
    limit: int = Field(
        default=25,
        description="Maximum number of pages to return (default: 25, max: 250)",
        ge=1,
        le=250,
    )
    status: str = Field(
        default="current",
        description="Filter by page status: 'current' (published), 'draft', 'archived', or 'any' (all statuses). Default: 'current'",
    )


class GetConfluenceSpacePagesTool:
    """Tool for listing pages in a Confluence space."""

    name = "Get Confluence Space Pages"
    description = """List all pages within a specific Confluence space.

    Use this tool when you need to:
    - Browse all documentation pages in a space
    - Discover the structure of documentation in a space
    - Find page hierarchies (parent-child relationships)
    - See which pages exist before creating new ones
    - List draft or archived pages in a space

    IMPORTANT: Requires the numeric space_id (NOT the space key).
    - First use 'Get Confluence Spaces' to get the space 'id' field
    - The 'id' is numeric like '1245186'
    - Do NOT use the 'key' field (like 'OOP' or '~712020...')

    This tool is useful for exploring a space's content, while 'Search Confluence Pages'
    is better for finding specific pages by keywords or criteria.

    After getting the page list:
    - Use 'Get Confluence Page' to read full content of a specific page
    - Use 'Create Confluence Page' to add new pages to this space

    Returns:
    - List of pages with title, ID, status, parent relationships, version, dates
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
        limit: int = 25,
        status: str = "current",
    ) -> Dict[str, Any]:
        """
        Get pages from a Confluence space.

        Args:
            space_id: The space ID
            limit: Maximum number of pages
            status: Filter by status

        Returns:
            Dictionary with success status and pages data
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
                # Get space pages
                status_param = None if status == "any" else status
                response = await client.get_space_pages(
                    space_id=space_id,
                    limit=limit,
                    status=status_param,
                )

                # Extract page information
                pages = []
                for page in response.get("results", []):
                    page_info = {
                        "id": page.get("id"),
                        "status": page.get("status"),
                        "title": page.get("title"),
                        "space_id": page.get("spaceId"),
                        "parent_id": page.get("parentId"),
                        "parent_type": page.get("parentType"),
                        "position": page.get("position"),
                        "author_id": page.get("authorId"),
                        "created_at": page.get("createdAt"),
                        "version": page.get("version", {}).get("number"),
                    }
                    pages.append(page_info)

                return {
                    "success": True,
                    "space_id": space_id,
                    "pages": pages,
                    "total": len(pages),
                    "status_filter": status,
                    "has_more": "_links" in response and "next" in response["_links"],
                }
            finally:
                await client.close()

        except Exception as e:
            logger.error(f"Error getting space pages: {str(e)}")
            return {"success": False, "error": str(e)}

    def run(
        self,
        space_id: str,
        limit: int = 25,
        status: str = "current",
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(space_id, limit, status))


def get_confluence_space_pages_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for getting pages in a Confluence space with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Confluence integration

    Returns:
        A configured StructuredTool for getting space pages
    """
    tool_instance = GetConfluenceSpacePagesTool(db, user_id)

    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Confluence Space Pages",
        description="""List all pages within a specific Confluence space.

        Use this when you need to:
        - Browse all pages in a space to understand its structure
        - Find page hierarchies (parent-child relationships)
        - Discover existing pages before creating new ones
        - List draft or archived pages

        Inputs:
        - space_id (str): The numeric space ID (NOT the key). Get the 'id' field from 'Get Confluence Spaces' tool output.
        - limit (int): Maximum pages to return (default: 25, max: 250)
        - status (str): Filter by 'current' (published), 'draft', 'archived', or 'any' (default: 'current')

        CRITICAL: Must use the space 'id' (numeric like '1245186'), NOT the 'key' (like 'OOP' or '~712020...').
        First call 'Get Confluence Spaces' and use the 'id' field from the returned spaces.

        This tool lists ALL pages in a space, while 'Search Confluence Pages' finds specific pages by keywords.

        After getting pages:
        - Use 'Get Confluence Page' to read full content of a specific page (using page_id)
        - Use 'Create Confluence Page' to add new documentation to this space

        Returns list of pages with title, ID, status, parent info, position, version, dates.""",
        args_schema=GetConfluenceSpacePagesInput,
    )
