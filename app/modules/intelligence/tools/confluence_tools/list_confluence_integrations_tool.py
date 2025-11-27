"""List Confluence Integrations Tool

Allows agents to list all available Confluence integrations for a user.
This is essential when a user has multiple Confluence workspaces connected.
"""

import logging
import asyncio
from typing import Any, Dict
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .confluence_client import get_available_confluence_integrations


class ListConfluenceIntegrationsInput(BaseModel):
    """Input schema for list_confluence_integrations tool."""

    pass  # No input parameters needed


class ListConfluenceIntegrationsTool:
    """Tool for listing all available Confluence integrations."""

    name = "List Confluence Integrations"
    description = """List all available Confluence integrations/workspaces connected by the user.

    **IMPORTANT: Call this tool FIRST when:**
    - User mentions a specific Confluence workspace/site name
    - Any Confluence tool returns an error about multiple integrations
    - You need to determine which Confluence integration to use

    **Purpose:**
    This tool helps identify which Confluence workspace to use when the user has
    multiple Confluence integrations connected (e.g., work Confluence, personal Confluence,
    different organizations, etc.).

    **Returns:**
    List of available integrations with:
    - integration_id: UUID to use in other Confluence tools (REQUIRED - use THIS, not site_name!)
    - site_name: Human-readable name (for display only, DO NOT use as integration_id)
    - site_url: Full URL of the Confluence workspace
    - created_at: When this integration was connected

    **CRITICAL:** Always use the 'integration_id' UUID field (e.g., 'da376af7-bff5-4c2f-a5da-0398de3601a8'),
    NEVER use the 'site_name' field (e.g., 'spoo') when calling other Confluence tools!

    **Workflow:**
    1. Call this tool to get all available Confluence integrations
    2. If user mentioned a site name, match it to the integration_id
    3. Use the integration_id in subsequent Confluence tool calls

    **Example:**
    User says: "Search Company A's Confluence for API docs"
    → Call list_confluence_integrations
    → Find integration where site_name contains "Company A"
    → Use that integration_id in search_confluence_pages tool
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

    async def arun(self) -> Dict[str, Any]:
        """
        List all available Confluence integrations for the user.

        Returns:
            Dictionary with success status and list of integrations
        """
        try:
            integrations = await get_available_confluence_integrations(
                self.user_id, self.db
            )

            if not integrations:
                return {
                    "success": True,
                    "count": 0,
                    "integrations": [],
                    "message": "No Confluence integrations found. Please connect your Confluence account in the Integrations page.",
                }

            return {
                "success": True,
                "count": len(integrations),
                "integrations": integrations,
                "message": f"Found {len(integrations)} Confluence integration(s). IMPORTANT: Use the 'integration_id' UUID field (NOT 'site_name') in other Confluence tools.",
            }

        except Exception as e:
            logging.error(f"Error listing Confluence integrations: {str(e)}")
            return {
                "success": False,
                "count": 0,
                "integrations": [],
                "error": str(e),
            }

    def run(self) -> Dict[str, Any]:
        """Synchronous version that runs the async version."""
        return asyncio.run(self.arun())


def list_confluence_integrations_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for listing Confluence integrations with user context.

    Args:
        db: Database session for integration retrieval
        user_id: The user ID to fetch their Confluence integrations

    Returns:
        A configured StructuredTool for listing Confluence integrations
    """
    tool_instance = ListConfluenceIntegrationsTool(db, user_id)

    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="List Confluence Integrations",
        description="""List all Confluence workspaces/integrations connected by the user.

        **CRITICAL: Call this tool FIRST when:**
        - User mentions a specific Confluence workspace name (e.g., "Company A's Confluence")
        - Any Confluence operation fails with "multiple integrations" error
        - You're unsure which Confluence to use

        **Purpose:**
        When users have multiple Confluence workspaces connected (work, personal, different orgs),
        this tool shows all available options so you can select the correct one.

        **No inputs required** - automatically retrieves all integrations for the current user.

        **Returns:**
        - count: Number of integrations found
        - integrations: Array of integration objects with:
          * integration_id: UUID (ALWAYS use THIS field in other Confluence tools - it's the actual ID!)
          * site_name: Human-readable workspace name (DISPLAY ONLY - never use as integration_id!)
          * site_url: Full Confluence URL
          * created_at: Connection date

        **CRITICAL WARNING:**
        When user mentions a workspace name (e.g., "spoo", "work"), you MUST:
        1. Match the name to find the correct integration object
        2. Extract the 'integration_id' UUID from that object
        3. Use that UUID in subsequent tool calls
        DO NOT pass the site_name itself - it will fail!

        **Next Steps:**
        1. If user mentioned a site name, match it to find the integration_id
        2. Pass the integration_id to other Confluence tools (search, get page, create page, etc.)
        3. If only one integration exists, you can use it automatically

        **Example Flow:**
        User: "Search my work Confluence for API docs"
        1. Call list_confluence_integrations
        2. Response: [{"integration_id": "da376af7-bff5-4c2f-a5da-0398de3601a8", "site_name": "Work Confluence"}, ...]
        3. Match "work" to find the object with site_name containing "Work"
        4. Extract integration_id = "da376af7-bff5-4c2f-a5da-0398de3601a8" (NOT "Work Confluence"!)
        5. Call search_confluence_pages(integration_id="da376af7-bff5-4c2f-a5da-0398de3601a8", cql="text ~ 'API'")

        **WRONG:** search_confluence_pages(integration_id="Work Confluence", ...)  ❌
        **RIGHT:** search_confluence_pages(integration_id="da376af7-bff5-4c2f-a5da-0398de3601a8", ...)  ✅""",
        args_schema=ListConfluenceIntegrationsInput,
    )
