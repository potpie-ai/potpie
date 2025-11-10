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
            "CQL (Confluence Query Language) query string.\n\n"
            "ESSENTIAL PATTERNS:\n"
            "Text Search (searches title, body, labels):\n"
            '  - text ~ "authentication"\n'
            '  - text ~ "API documentation"\n'
            '  - space=DEV AND text ~ "deploy"\n\n'
            "Filter by Space:\n"
            "  - type=page AND space=DEMO\n"
            "  - type=page AND space in (DEV,PROD)\n\n"
            "Search by Title:\n"
            '  - title ~ "API"\n\n'
            "Filter by Label:\n"
            "  - label=api\n"
            "  - label in (backend,frontend)\n\n"
            "Combine Filters:\n"
            '  - type=page AND space=DEV AND text ~ "authentication"\n'
            '  - space=PROD AND label=api AND text ~ "endpoint"'
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
    
    PRIMARY tool for finding documentation in Confluence.
    
    Most common use: text ~ "keyword" to search across titles and page content.
    
    Examples:
    - text ~ "authentication" - Find pages about authentication
    - space=DEV AND text ~ "API" - Search within DEV space
    - type=page AND space in (DEV,PROD) - Pages in multiple spaces
    - title ~ "getting started" - Search by title
    - label=api - Pages tagged with 'api' label
    - space=PROD AND label=backend - Combine space and label filters
    
    After finding pages, use 'Get Confluence Page' to read full content.
    
    Returns: List of pages with title, ID, space, URL, excerpt
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

            # Extract page results from v1 API response
            # v1 API returns results with a different structure than v2
            pages = []
            for result in response.get("results", []):
                # In v1, content info is in result.content
                content = result.get("content", {})

                page_info = {
                    "id": content.get("id"),
                    "status": content.get("status"),
                    "title": result.get("title") or content.get("title"),
                    "url": result.get("url"),
                    "excerpt": result.get("excerpt", ""),
                    "last_modified": result.get("lastModified"),
                    "entity_type": result.get("entityType"),
                }

                # Add space info if available
                space = result.get("space") or content.get("space", {})
                if space:
                    page_info["space_key"] = space.get("key")
                    page_info["space_name"] = space.get("name")

                # Add version info if available
                version = content.get("version", {})
                if version:
                    page_info["version"] = version.get("number")

                pages.append(page_info)

            client.close()

            return {
                "success": True,
                "pages": pages,
                "total": response.get("totalSize", len(pages)),
                "limit": response.get("limit", limit),
                "has_more": response.get("size", 0) < response.get("totalSize", 0),
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
        description="""Search Confluence for documentation using CQL (Confluence Query Language).
        
        Primary use: Finding documentation by searching text content.
        
        Common patterns:
        • text ~ "keyword" - Search page content
        • space=DEV AND text ~ "API" - Search in specific space
        • type=page AND space in (DEV,PROD) - Multiple spaces
        • title ~ "getting started" - Search titles
        • label=api - Pages with 'api' label
        • space=PROD AND label=backend - Combine filters
        
        The 'text' field searches across page titles, body content, and labels.
        
        Inputs:
        - cql: CQL query string
        - limit: Max results (default: 25, max: 250)
        - include_archived: Include archived pages (default: False)
        
        Returns: Pages with title, id, space_key, space_name, url, excerpt
        
        Next step: Use 'Get Confluence Page' with the page id to read full content.""",
        args_schema=SearchConfluencePagesInput,
    )
