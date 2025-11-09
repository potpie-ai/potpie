"""Search Confluence Pages Tool

Allows agents to search for Confluence pages using CQL (Confluence Query Language).
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


class SearchConfluencePagesInput(BaseModel):
    """Input schema for search_confluence_pages tool."""

    cql: str = Field(
        description=(
            "CQL (Confluence Query Language) query string to search pages. Examples:\n"
            "- 'type=page AND title~\"API\"' - Search pages with 'API' in title\n"
            "- 'space=DEMO AND text~\"documentation\"' - Search text in DEMO space\n"
            "- 'type=page AND label=\"api\"' - Find pages with 'api' label\n"
            "- 'creator=currentUser() ORDER BY created DESC' - Your recent pages\n"
            "- 'type=page AND created >= \"2024-01-01\"' - Pages created this year\n"
            "- 'type=page AND space in (DEMO,PROD)' - Search multiple spaces"
        )
    )
    limit: int = Field(
        default=25,
        description="Maximum number of results to return (default: 25, max: 250)",
        ge=1,
        le=250,
    )
    include_archived: bool = Field(
        default=False,
        description="If True, include archived pages in search results. Default: False",
    )


class SearchConfluencePagesTool:
    """Tool for searching Confluence pages using CQL."""

    name = "Search Confluence Pages"
    description = """Search for Confluence pages using CQL (Confluence Query Language).
    
    This is the PRIMARY tool for finding relevant documentation in Confluence.
    
    Use this tool when you need to:
    - Find pages by title, content, or keywords
    - Search for documentation in specific spaces
    - Find pages created or updated within a time range
    - Search by labels, creator, or other metadata
    - Discover pages related to a topic
    
    CQL Query Examples:
    - 'type=page AND title~"API Documentation"' - Find pages with API Documentation in title
    - 'space=DEMO AND text~"authentication"' - Search for authentication in DEMO space
    - 'type=page AND label="backend"' - Find pages tagged with backend
    - 'creator=currentUser() ORDER BY created DESC' - Your recent pages
    - 'type=page AND created >= "2024-01-01"' - Pages from this year
    - 'type=page AND space in (DEMO,PROD) AND text~"deploy"' - Search multiple spaces
    
    After finding pages:
    - Use 'Get Confluence Page' to read full content (using page_id from results)
    - Use 'Update Confluence Page' to modify content
    - Use 'Add Confluence Comment' to comment
    
    Returns:
    - List of matching pages with title, ID, space, author, creation date, excerpt
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
        cql: str,
        limit: int = 25,
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        """
        Search Confluence pages using CQL.

        Args:
            cql: CQL query string
            limit: Maximum number of results
            include_archived: Include archived pages

        Returns:
            Dictionary with success status and search results
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

            # Search pages
            response = client.search_pages(
                cql=cql,
                limit=limit,
                include_archived=include_archived,
            )

            # Extract page results
            pages = []
            for page in response.get("results", []):
                page_info = {
                    "id": page.get("id"),
                    "status": page.get("status"),
                    "title": page.get("title"),
                    "space_id": page.get("spaceId"),
                    "parent_id": page.get("parentId"),
                    "parent_type": page.get("parentType"),
                    "author_id": page.get("authorId"),
                    "created_at": page.get("createdAt"),
                    "version": page.get("version", {}).get("number"),
                }

                # Add excerpt if available
                body = page.get("body", {})
                if "storage" in body:
                    content = body["storage"].get("value", "")
                    # Get first 200 chars as excerpt
                    page_info["excerpt"] = (
                        content[:200] + "..." if len(content) > 200 else content
                    )

                pages.append(page_info)

            client.close()

            return {
                "success": True,
                "pages": pages,
                "total": len(pages),
                "has_more": "_links" in response and "next" in response["_links"],
                "query": cql,
            }

        except Exception as e:
            logging.error(f"Error searching Confluence pages: {str(e)}")
            return {"success": False, "error": str(e)}

    def run(
        self,
        cql: str,
        limit: int = 25,
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun(cql, limit, include_archived))


def search_confluence_pages_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for searching Confluence pages with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their specific Confluence integration

    Returns:
        A configured StructuredTool for searching Confluence pages
    """
    tool_instance = SearchConfluencePagesTool(db, user_id)

    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Search Confluence Pages",
        description="""Search for Confluence pages using CQL (Confluence Query Language).
        
        PRIMARY tool for finding documentation in Confluence.
        
        Use this when you need to:
        - Find pages by title, content, or keywords
        - Search specific spaces or across all spaces
        - Find recent pages or pages in a date range
        - Search by labels, creator, or metadata
        
        Inputs:
        - cql (str): CQL query (e.g., 'type=page AND text~"keyword"', 'space=DEMO AND title~"API"')
        - limit (int): Maximum results (default: 25, max: 250)
        - include_archived (bool): Include archived pages (default: False)
        
        CQL Tips:
        - Use ~ for contains matching: title~"API"
        - Use = for exact matching: space=DEMO
        - Combine with AND, OR: space=DEMO AND label="api"
        - Order results: ORDER BY created DESC
        - Date filters: created >= "2024-01-01"
        
        After finding pages, use 'Get Confluence Page' to read full content (needs page_id from results).
        
        Returns list of pages with title, ID, space, author, excerpt, and metadata.""",
        args_schema=SearchConfluencePagesInput,
    )
