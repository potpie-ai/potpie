import asyncio
from typing import Dict, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.parsing.knowledge_graph.inference_schema import QueryResponse
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.projects.projects_service import ProjectService


class QueryRequest(BaseModel):
    node_ids: List[str] = Field(description="A list of node ids to query")
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )
    query: str = Field(
        description="A natural language question to ask the knowledge graph"
    )


class MultipleKnowledgeGraphQueriesInput(BaseModel):
    queries: List[str] = Field(
        description="A list of natural language questions to ask the knowledge graph"
    )
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )


class KnowledgeGraphQueryTool:
    name = "Ask Knowledge Graph Queries"
    description = """Query the code knowledge graph using natural language questions.
    The knowledge graph contains information about every function, class, and file in the codebase.
    This tool allows asking multiple questions about the codebase in a single operation.
      Use this tool when you need to ask multiple related questions about the codebase at once.
    Do not use this to query code directly. The inputs structure is as foillowing
        :param queries: array, list of natural language questions to ask about the codebase.
        :param project_id: string, the project ID (UUID).
        :param node_ids: array, optional list of node IDs to query (use when answer relates to specific nodes).

            example:
            {
                "queries": ["What does the UserService class do?", "How is authentication implemented?"],
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_ids": ["123e4567-e89b-12d3-a456-426614174000"]
            }

        Returns list of query responses with relevant code information.
        """

    def __init__(self, sql_db, user_id):
        self.headers = {"Content-Type": "application/json"}
        self.user_id = user_id
        self.sql_db = sql_db

    async def ask_multiple_knowledge_graph_queries(
        self, queries: List[QueryRequest]
    ) -> Dict[str, str]:
        inference_service = InferenceService(self.sql_db, "dummy")

        async def process_query(query_request: QueryRequest) -> List[QueryResponse]:
            # Call the query_vector_index method directly from InferenceService
            results = inference_service.query_vector_index(
                query_request.project_id, query_request.query, query_request.node_ids
            )
            return [
                QueryResponse(
                    node_id=result.get("node_id"),
                    docstring=result.get("docstring"),
                    file_path=result.get("file_path"),
                    start_line=result.get("start_line") or 0,
                    end_line=result.get("end_line") or 0,
                    similarity=result.get("similarity"),
                )
                for result in results
            ]

        tasks = [process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)

        return results

    async def arun(
        self, queries: List[str], project_id: str, node_ids: List[str] = []
    ) -> Dict[str, str]:
        return await asyncio.to_thread(self.run, queries, project_id, node_ids)

    def run(
        self, queries: List[str], project_id: str, node_ids: List[str] = []
    ) -> Dict[str, str]:
        """
        Query the code knowledge graph using multiple natural language questions.
        The knowledge graph contains information about every function, class, and file in the codebase.
        This method allows asking multiple questions about the codebase in a single operation.

        Inputs:
        - queries (List[str]): A list of natural language questions that the user wants to ask the knowledge graph.
          Each question should be clear and concise, related to the codebase.
        - project_id (str): The ID of the project being evaluated, this is a UUID.
        - node_ids (List[str]): A list of node ids to query, this is an optional parameter that can be used to query a specific node.

        Returns:
        - Dict[str, str]: A dictionary where keys are the original queries and values are the corresponding responses.
        """
        project = asyncio.run(
            ProjectService(self.sql_db).get_project_repo_details_from_db(
                project_id, self.user_id
            )
        )
        if not project:
            raise ValueError(
                f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
            )
        project_id = project["id"]
        query_list = [
            QueryRequest(query=query, project_id=project_id, node_ids=node_ids)
            for query in queries
        ]
        return asyncio.run(self.ask_multiple_knowledge_graph_queries(query_list))


def get_ask_knowledge_graph_queries_tool(sql_db, user_id) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=KnowledgeGraphQueryTool(sql_db, user_id).arun,
        func=KnowledgeGraphQueryTool(sql_db, user_id).run,
        name="Ask Knowledge Graph Queries",
        description="""
    Query the code knowledge graph using multiple natural language questions.
    The knowledge graph contains information about every function, class, and file in the codebase.
    This tool allows asking multiple questions about the codebase in a single operation.

    Inputs:
    - queries (List[str]): A list of natural language questions to ask the knowledge graph. Each question should be
    clear and concise, related to the codebase, such as "What does the XYZ class do?" or "How is the ABC function used?"
    - project_id (str): The ID of the project being evaluated, this is a UUID.
    - node_ids (List[str]): A list of node ids to query, this is an optional parameter that can be used to query a specific node. use this only when you are sure that the answer to the question is related to that node.

    Use this tool when you need to ask multiple related questions about the codebase at once.
    Do not use this to query code directly.""",
        args_schema=MultipleKnowledgeGraphQueriesInput,
    )
