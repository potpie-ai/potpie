"""Get Confluence Spaces Tool

Allows agents to fetch list of available Confluence spaces for the user.
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


class GetConfluenceSpacesInput(BaseModel):
    """Input schema for get_confluence_spaces tool."""

    limit: int = Field(
        default=25,
        description="Maximum number of spaces to return (default: 25, max: 250)",
        ge=1,
        le=250,
    )
    space_type: str = Field(
        default="all",
        description="Filter by space type: 'global' (team spaces), 'personal' (personal spaces), or 'all' (both). Default: 'all'",
    )


class GetConfluenceSpacesTool:
    """Tool for fetching Confluence spaces."""

    name = "Get Confluence Spaces"
    description = """Get list of Confluence spaces available to the user.

    Use this tool when you need to:
    - Discover available Confluence spaces before creating or searching for pages
    - Find the space ID or key needed for other operations
    - List all spaces accessible to the user
    - Filter spaces by type (global team spaces vs personal spaces)

    This is typically the FIRST tool to use when working with Confluence - you need
    to know which spaces exist before you can create pages or search for content.

    Once you have a space_id, you can:
    - Use 'Get Confluence Space Pages' to list all pages in that space
    - Use 'Create Confluence Page' to add new documentation to that space
    - Use 'Search Confluence Pages' with space filter in CQL query

    Returns:
    - List of spaces with ID, key, name, type, status, and description
    - Total count and pagination info
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

    async def arun(self, limit: int = 25, space_type: str = "all") -> Dict[str, Any]:
        """
        Get list of Confluence spaces.

        Args:
            limit: Maximum number of spaces to return
            space_type: Filter by type (global, personal, all)

        Returns:
            Dictionary with success status and spaces data
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
                # Get spaces
                space_type_param = None if space_type == "all" else space_type
                response = await client.get_spaces(limit=limit, type=space_type_param)

                spaces = []
                for space in response.get("results", []):
                    # Handle description which can be null or an object
                    description = ""
                    desc_obj = space.get("description")
                    if desc_obj and isinstance(desc_obj, dict):
                        plain_obj = desc_obj.get("plain", {})
                        if plain_obj and isinstance(plain_obj, dict):
                            description = plain_obj.get("value", "")

                    space_info = {
                        "id": space.get("id"),
                        "key": space.get("key"),
                        "name": space.get("name"),
                        "type": space.get("type"),
                        "status": space.get("status"),
                        "homepage_id": space.get("homepageId"),
                        "description": description,
                    }
                    spaces.append(space_info)

                return {
                    "success": True,
                    "spaces": spaces,
                    "total": len(spaces),
                    "has_more": "_links" in response and "next" in response["_links"],
                }
            finally:
                await client.close()

        except Exception as e:
            logging.error(f"Error getting Confluence spaces: {str(e)}")
            return {"success": False, "error": str(e)}

    def run(self, limit: int = 25, space_type: str = "all") -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(limit, space_type))


def get_confluence_spaces_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for getting Confluence spaces with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Confluence integration

    Returns:
        A configured StructuredTool for getting Confluence spaces
    """
    tool_instance = GetConfluenceSpacesTool(db, user_id)

    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Confluence Spaces",
        description="""Get list of Confluence spaces available to the user.

        Use this as the FIRST step when working with Confluence to discover available spaces.

        Use this when you need to:
        - Find which spaces exist and are accessible
        - Get space ID (required for listing/creating pages) or key (used in URLs)
        - Distinguish between global (team) and personal spaces

        Inputs:
        - limit (int): Maximum spaces to return (default: 25, max: 250)
        - space_type (str): Filter by 'global', 'personal', or 'all' (default: 'all')

        IMPORTANT: Returns space 'id' field which is REQUIRED for:
        - 'Get Confluence Space Pages' (needs space_id parameter)
        - 'Create Confluence Page' (needs space_id parameter)

        Also use 'Search Confluence Pages' with space filter to find content.

        Returns list of spaces with ID (numeric), key (string identifier), name, type, and description.""",
        args_schema=GetConfluenceSpacesInput,
    )
