"""Confluence Tools

Collection of tools for interacting with Confluence API.
These tools enable AI agents to search, read, create, update, and comment on Confluence pages.
"""

from sqlalchemy.orm import Session
from app.modules.intelligence.tools.tool_schema import OnyxTool

from .get_confluence_spaces_tool import get_confluence_spaces_tool
from .get_confluence_page_tool import get_confluence_page_tool
from .search_confluence_pages_tool import search_confluence_pages_tool
from .get_confluence_space_pages_tool import get_confluence_space_pages_tool
from .create_confluence_page_tool import create_confluence_page_tool
from .update_confluence_page_tool import update_confluence_page_tool
from .add_confluence_comment_tool import add_confluence_comment_tool

from .confluence_client import check_confluence_integration_exists


def get_all_confluence_tools(db: Session, user_id: str) -> list[OnyxTool]:
    """
    Get all Confluence tools for a user.

    Args:
        db: Database session
        user_id: The user ID

    Returns:
        List of all Confluence tools
    """
    return [
        get_confluence_spaces_tool(db, user_id),
        get_confluence_page_tool(db, user_id),
        search_confluence_pages_tool(db, user_id),
        get_confluence_space_pages_tool(db, user_id),
        create_confluence_page_tool(db, user_id),
        update_confluence_page_tool(db, user_id),
        add_confluence_comment_tool(db, user_id),
    ]


__all__ = [
    "get_confluence_spaces_tool",
    "get_confluence_page_tool",
    "search_confluence_pages_tool",
    "get_confluence_space_pages_tool",
    "create_confluence_page_tool",
    "update_confluence_page_tool",
    "add_confluence_comment_tool",
    "get_all_confluence_tools",
    "check_confluence_integration_exists",
]
