"""Get Confluence Page Tool

Allows agents to fetch content of a specific Confluence page by ID.
"""

import logging
import asyncio
from typing import Any, Dict
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .confluence_client import (
    get_confluence_client_for_user,
    check_confluence_integration_exists,
)


class GetConfluencePageInput(BaseModel):
    """Input schema for get_confluence_page tool."""

    page_id: str = Field(
        description="The ID of the Confluence page to retrieve (e.g., '123456'). Use 'Search Confluence Pages' or 'Get Confluence Space Pages' to find page IDs."
    )
    body_format: str = Field(
        default="storage",
        description="Format for page body: 'storage' (HTML - recommended), 'atlas_doc_format' (ADF JSON), or 'view' (rendered HTML). Default: 'storage'",
    )
    get_draft: bool = Field(
        default=False,
        description="If True, retrieve the draft version instead of the published version. Default: False",
    )


class GetConfluencePageTool:
    """Tool for fetching a specific Confluence page."""

    name = "Get Confluence Page"
    description = """Fetch detailed content and metadata of a specific Confluence page by ID.

    Use this tool when you need to:
    - Read the full content of a specific documentation page
    - Get page metadata (author, version, labels, parent page)
    - Check the current status of a page (published/draft/archived)
    - Retrieve page content before updating it (to get version number)
    - Extract information from a known page

    IMPORTANT: You need the page_id to use this tool. Get it from:
    - 'Search Confluence Pages' tool (search by title or content)
    - 'Get Confluence Space Pages' tool (browse pages in a space)

    Once you have the page data:
    - Use 'Update Confluence Page' to modify the content (requires version number from this tool)
    - Use 'Add Confluence Comment' to comment on the page

    Returns:
    - Page title, content, author, version info, labels, space ID, parent page info
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
        body_format: str = "storage",
        get_draft: bool = False,
    ) -> Dict[str, Any]:
        """
        Get a Confluence page by ID.

        Args:
            page_id: The page ID
            body_format: Format for page body
            get_draft: Whether to get draft version

        Returns:
            Dictionary with success status and page data
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
                # Get page
                page = await client.get_page(
                    page_id=page_id,
                    body_format=body_format,
                    get_draft=get_draft,
                )

                # Extract page information
                page_data = {
                    "id": page.get("id"),
                    "status": page.get("status"),
                    "title": page.get("title"),
                    "space_id": page.get("spaceId"),
                    "parent_id": page.get("parentId"),
                    "parent_type": page.get("parentType"),
                    "position": page.get("position"),
                    "author_id": page.get("authorId"),
                    "created_at": page.get("createdAt"),
                    "version": {
                        "number": page.get("version", {}).get("number"),
                        "message": page.get("version", {}).get("message", ""),
                        "created_at": page.get("version", {}).get("createdAt"),
                    },
                }

                # Extract body content based on format
                body = page.get("body", {})
                if body_format == "storage" and "storage" in body:
                    page_data["content"] = body["storage"].get("value", "")
                    page_data["content_format"] = "storage"
                elif body_format == "atlas_doc_format" and "atlas_doc_format" in body:
                    page_data["content"] = body["atlas_doc_format"].get("value", "")
                    page_data["content_format"] = "atlas_doc_format"
                elif body_format == "view" and "view" in body:
                    page_data["content"] = body["view"].get("value", "")
                    page_data["content_format"] = "view"
                else:
                    page_data["content"] = ""
                    page_data["content_format"] = body_format

                # Extract labels
                labels = []
                for label in page.get("labels", {}).get("results", []):
                    labels.append(
                        {
                            "id": label.get("id"),
                            "name": label.get("name"),
                            "prefix": label.get("prefix"),
                        }
                    )
                page_data["labels"] = labels

                return {
                    "success": True,
                    "page": page_data,
                }
            finally:
                await client.close()

        except Exception as e:
            logging.error(f"Error getting Confluence page: {str(e)}")
            return {"success": False, "error": str(e)}

    def run(
        self,
        page_id: str,
        body_format: str = "storage",
        get_draft: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(page_id, body_format, get_draft))


def get_confluence_page_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for getting a Confluence page with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Confluence integration

    Returns:
        A configured StructuredTool for getting Confluence pages
    """
    tool_instance = GetConfluencePageTool(db, user_id)

    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Confluence Page",
        description="""Get content and metadata of a specific Confluence page by ID.

        Use this when you need to:
        - Read the full content of a documentation page
        - Get page metadata (title, author, version, labels, dates)
        - Retrieve page before updating (to get current version number)
        - Check page status (published/draft/archived)

        Inputs:
        - page_id (str): The page ID (get from 'Search Confluence Pages' or 'Get Confluence Space Pages')
        - body_format (str): Content format - 'storage' (HTML, default), 'atlas_doc_format' (ADF), or 'view' (rendered)
        - get_draft (bool): Get draft version instead of published (default: False)

        IMPORTANT: To update a page, you MUST use this tool first to get the current version number.

        After getting page:
        - Use 'Update Confluence Page' to modify content (requires version.number from this response)
        - Use 'Add Confluence Comment' to comment on the page

        Returns page with title, content, author, version info, labels, space ID, parent info.""",
        args_schema=GetConfluencePageInput,
    )
