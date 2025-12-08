import asyncio
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.projects.projects_service import ProjectService


class NodeSearchInput(BaseModel):
    query: str = Field(
        description="A natural language search query to find matching code elements (functions, classes, files)"
    )
    project_id: str = Field(description="The project ID (UUID) to search within")
    top_k: int = Field(
        default=5, description="Maximum number of results to return (default: 5)"
    )


class NodeSearchResult(BaseModel):
    node_id: str
    docstring: str
    file_path: str
    start_line: int
    end_line: int
    similarity: float


class NodeSearchFromTextTool:
    name = "Search Code Docstring"
    description = """Search the codebase using natural language to find matching functions, classes, or files.
    This tool performs semantic search over the code knowledge graph embeddings.

    Use this tool when you need to:
    - Find code elements by describing what they do (e.g., "user authentication logic")
    - Locate functions or classes related to a concept (e.g., "database connection handling")
    - Discover relevant code without knowing exact names

    Inputs:
    - query (str): A short natural language description of the code action you're looking for.
      Examples: "validates email addresses", "payment processing"
    - project_id (str): The project ID (UUID) to search within.
    - top_k (int): Maximum number of results to return (default: 5).

    Returns a list of matching code elements with:
    - node_id: Unique identifier for the code element
    - docstring: Description of what the code does
    - file_path: Path to the file containing the code
    - start_line/end_line: Line numbers where the code is located
    - similarity: Relevance score (higher is better)

    example:
    {
        "query": "dependency comparison",
        "project_id": "550e8400-e29b-41d4-a716-446655440000",
        "top_k": 10
    }
    """

    def __init__(self, sql_db, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    async def arun(
        self, query: str, project_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the codebase using natural language query asynchronously."""
        return await asyncio.to_thread(self.run, query, project_id, top_k)

    def run(self, query: str, project_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the codebase using natural language to find matching code elements.

        Args:
            query: Natural language description of what you're looking for
            project_id: The project ID (UUID) to search within
            top_k: Maximum number of results to return

        Returns:
            List of matching code elements with node_id, docstring, file_path,
            start_line, end_line, and similarity score
        """
        # Validate project exists and user has access
        project = asyncio.run(
            ProjectService(self.sql_db).get_project_repo_details_from_db(
                project_id, self.user_id
            )
        )
        if not project:
            raise ValueError(
                f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
            )

        validated_project_id = project["id"]

        inference_service = InferenceService(self.sql_db, self.user_id)
        try:
            results = inference_service.query_vector_index(
                project_id=validated_project_id,
                query=query,
                node_ids=None,  # Search across all nodes
                top_k=top_k,
            )

            if not results:
                return [
                    {"message": f"No matching code elements found for query: '{query}'"}
                ]

            return [
                {
                    "node_id": result.get("node_id"),
                    "docstring": result.get("docstring"),
                    "file_path": result.get("file_path"),
                    "start_line": result.get("start_line") or 0,
                    "end_line": result.get("end_line") or 0,
                    "similarity": result.get("similarity"),
                }
                for result in results
            ]
        finally:
            inference_service.close()


def node_search_from_text_tool(sql_db, user_id: str) -> StructuredTool:
    """Create and return the node search from text tool."""
    tool_instance = NodeSearchFromTextTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=NodeSearchInput,
    )
