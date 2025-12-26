import asyncio
from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.intelligence.memory.memory_service_factory import MemoryServiceFactory
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class SearchMemoryInput(BaseModel):
    """Input for memory search tool"""

    query: str = Field(
        description="What to search for in memories (e.g., 'coding preferences', 'ORM preferences', 'name')"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="The project ID (UUID) to search within. Leave empty to search user-level memories only.",
    )
    scope: Optional[str] = Field(
        default=None,
        description="Filter by memory scope: 'user' (available in all projects) or 'project' (this project only). Leave empty to search both.",
    )
    memory_type: Optional[str] = Field(
        default=None,
        description="Filter by memory type: 'semantic' (facts/preferences) or 'episodic' (events/actions). Leave empty to search both.",
    )
    limit: Optional[int] = Field(
        default=5, description="Maximum number of memories to return (default: 5)"
    )


class MemorySearchTool:
    """Tool for searching user memories and preferences"""

    name = "search_user_memories"
    description = """🧠 Search user memories and preferences with powerful filtering. Use this to recall:

**What you can search for:**
- User's coding style preferences (e.g., camelCase vs snake_case)
- Tool and framework preferences (e.g., which ORM they prefer)
- Project-specific decisions and architecture choices
- Personal information (name, timezone, work hours)
- Past events and actions (bugs fixed, features deployed)

**Parameters:**
- `query`: What to search for
- `project_id`: The project UUID to search within (required for project-scoped search)
- `scope`: "user" for cross-project preferences, "project" for this project only, or leave empty for both
- `memory_type`: "semantic" for facts/preferences, "episodic" for time-based events, or leave empty for both
- `limit`: Number of results to return (default: 5)

**Memory scopes:**
- 👤 User-level: Available across ALL projects (e.g., "Prefers camelCase")
- 📁 Project-level: Specific to THIS project only (e.g., "Project uses PostgreSQL")

**Memory types:**
- 💡 Semantic: Facts, preferences, and knowledge
- 📅 Episodic: Time-based events and actions

**Examples:**
- Search for coding preferences: query="coding style", scope="user"
- Search for project tech stack: query="database ORM", project_id="<uuid>", scope="project"
- Search for recent fixes: query="bug fixes", memory_type="episodic"

Use this proactively when you need context about user preferences or past decisions!"""

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.memory_service = MemoryServiceFactory.create()
        self._closed = False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.close()
        return False

    def close(self):
        """Close the memory service and release resources"""
        if not self._closed and self.memory_service:
            try:
                self.memory_service.close()
            except Exception as e:
                logger.warning(f"Error closing memory service: {e}")
            finally:
                self._closed = True

    def __del__(self):
        """Fallback cleanup when instance is garbage collected"""
        if not self._closed and hasattr(self, "memory_service") and self.memory_service:
            try:
                self.memory_service.close()
            except Exception:
                # Ignore errors during garbage collection
                pass

    def _format_search_results(
        self,
        search_response,
        query: str,
        scope: Optional[str],
        memory_type: Optional[str],
    ) -> str:
        """
        Format search results into a readable string

        Args:
            search_response: The search response object with results
            query: The search query
            scope: The scope filter that was applied
            memory_type: The memory type filter that was applied

        Returns:
            Formatted string with relevant memories, or message if none found
        """
        if not search_response.results:
            filters = []
            if scope:
                filters.append(f"scope={scope}")
            if memory_type:
                filters.append(f"type={memory_type}")
            filter_str = f" ({', '.join(filters)})" if filters else ""
            return f"No memories found for: {query}{filter_str}"

        # Format results
        filters = []
        if scope:
            filters.append(f"scope={scope}")
        if memory_type:
            filters.append(f"type={memory_type}")
        filter_str = f" ({', '.join(filters)})" if filters else ""

        results_text = (
            f"Found {len(search_response.results)} relevant memories{filter_str}:\n\n"
        )

        for i, result in enumerate(search_response.results, 1):
            result_scope = result.metadata.get("scope", "unknown")
            result_type = result.metadata.get("memory_type", "unknown")

            # Add scope indicator
            scope_icon = "👤" if result_scope == "user" else "📁"
            type_icon = "💡" if result_type == "semantic" else "📅"

            results_text += f"{i}. {scope_icon}{type_icon} {result.memory}\n"
            results_text += f"   (scope: {result_scope}, type: {result_type})\n\n"

        logger.info(
            f"[memory search tool] Returned {len(search_response.results)} memories"
        )

        return results_text.strip()

    async def arun(
        self,
        query: str,
        project_id: Optional[str] = None,
        scope: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Async version of memory search - directly calls _async_search without nested event loops

        Args:
            query: What to search for (e.g., "coding style preferences", "ORM choice")
            project_id: Project UUID to search within
            scope: Filter by scope - "user" for user-level, "project" for project-level, None for both
            memory_type: Filter by type - "semantic" for facts/preferences, "episodic" for events, None for both
            limit: Maximum number of memories to return

        Returns:
            Formatted string with relevant memories, or message if none found
        """
        try:
            logger.info(
                f"[memory search tool] Searching: query='{query}', "
                f"user_id={self.user_id}, project_id={project_id}, scope={scope}, memory_type={memory_type}, limit={limit}"
            )

            search_response = await self._async_search(
                query, project_id, scope, memory_type, limit
            )

            return self._format_search_results(
                search_response, query, scope, memory_type
            )

        except Exception as e:
            logger.error(f"[memory search tool] Error: {e}", exc_info=True)
            return f"Error searching memories: {str(e)}"

    def run(
        self,
        query: str,
        project_id: Optional[str] = None,
        scope: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Search user memories and preferences relevant to the current task

        Args:
            query: What to search for (e.g., "coding style preferences", "ORM choice")
            project_id: Project UUID to search within
            scope: Filter by scope - "user" for user-level, "project" for project-level, None for both
            memory_type: Filter by type - "semantic" for facts/preferences, "episodic" for events, None for both
            limit: Maximum number of memories to return

        Returns:
            Formatted string with relevant memories, or message if none found
        """
        try:
            logger.info(
                f"[memory search tool] Searching: query='{query}', "
                f"user_id={self.user_id}, project_id={project_id}, scope={scope}, memory_type={memory_type}, limit={limit}"
            )

            # Run async search in sync context
            search_response = asyncio.run(
                self._async_search(query, project_id, scope, memory_type, limit)
            )

            return self._format_search_results(
                search_response, query, scope, memory_type
            )

        except Exception as e:
            logger.error(f"[memory search tool] Error: {e}", exc_info=True)
            return f"Error searching memories: {str(e)}"

    async def _async_search(
        self,
        query: str,
        project_id: Optional[str],
        scope: Optional[str],
        memory_type: Optional[str],
        limit: int,
    ):
        """Internal async search method"""
        # Determine search strategy based on scope filter
        if scope == "user":
            # Search only user-level memories (project_id=None, include_user_preferences=True)
            return await self.memory_service.search(
                query=query,
                user_id=self.user_id,
                project_id=None,  # Force user-level agent
                limit=limit,
                include_user_preferences=True,
                memory_type=memory_type,
            )
        elif scope == "project":
            # Search only project-level memories (with project_id, include_user_preferences=False)
            if not project_id:
                raise ValueError(
                    "Cannot search project-level memories: project_id is required"
                )
            return await self.memory_service.search(
                query=query,
                user_id=self.user_id,
                project_id=project_id,
                limit=limit,
                include_user_preferences=False,  # Don't include user-level
                memory_type=memory_type,
            )
        else:
            # Search both user and project memories
            return await self.memory_service.search(
                query=query,
                user_id=self.user_id,
                project_id=project_id,
                limit=limit,
                include_user_preferences=True,
                memory_type=memory_type,
            )


def search_user_memories_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a memory search tool following the KG tool pattern

    Args:
        db: Database session
        user_id: User identifier for memory scoping

    Returns:
        StructuredTool that searches memories
    """
    tool_instance = MemorySearchTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="search_user_memories",
        description="""🧠 Search user memories and preferences with powerful filtering.

**What you can search for:**
- User's coding style preferences (e.g., camelCase vs snake_case)
- Tool and framework preferences (e.g., which ORM they prefer)
- Project-specific decisions and architecture choices
- Personal information (name, timezone, work hours)
- Past events and actions (bugs fixed, features deployed)

**Inputs:**
- query (str): What to search for
- project_id (str, optional): The project UUID to search within
- scope (str, optional): "user" for user-level, "project" for project-level, or empty for both
- memory_type (str, optional): "semantic" for facts/preferences, "episodic" for events, or empty for both
- limit (int, optional): Number of results (default: 5)

**Memory scopes:**
- 👤 User-level: Available across ALL projects (e.g., "Prefers camelCase")
- 📁 Project-level: Specific to THIS project only (e.g., "Project uses PostgreSQL")

**Memory types:**
- 💡 Semantic: Facts, preferences, and knowledge
- 📅 Episodic: Time-based events and actions

Use this proactively when you need context about user preferences or past decisions!""",
        args_schema=SearchMemoryInput,
    )
