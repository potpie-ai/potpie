import os
from typing import Type

import requests
from langchain.tools import BaseTool as LangchainToolBaseModel
from pydantic import BaseModel


class KnowledgeGraphQueryInput(BaseModel):
    query: str
    project_id: str


class KnowledgeGraphQueryTool(LangchainToolBaseModel):
    name = "KnowledgeGraphQuery"
    description = (
        "Query the code knowledge graph with specific directed questions using natural language "
        "and project id. This tool returns the query result. It should be used when you need "
        "to query the knowledge graph for information related to APIs, endpoints, or code explanations."
        "DO NOT USE THIS TOOL TO QUERY CODE DIRECTLY. USE GET CODE TOOL TO FETCH CODE. "
        "The knowledge graph contains information from various database tables including:\n"
        "- inference: key-value pairs of inferred knowledge about APIs and their constituting functions\n"
        "- endpoints: API endpoint paths and identifiers\n"
        "- explanation: code explanations for function identifiers\n"
        "- pydantic: pydantic class definitions\n"
    )
    args_schema: Type[BaseModel] = KnowledgeGraphQueryInput

    def _run(self, query: str, project_id: str) -> str:
        """
        Synchronously query the code knowledge graph using natural language questions.
        This method sends a query and project_id to the knowledge graph and returns the response.
        """
        data = {"project_id": project_id, "query": query}
        headers = {"Content-Type": "application/json"}
        kg_query_url = os.getenv("KNOWLEDGE_GRAPH_URL")
        if not kg_query_url:
            raise ValueError("KNOWLEDGE_GRAPH_URL environment variable is not set")

        try:
            response = requests.post(kg_query_url, json=data, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except requests.RequestException as e:
            return f"Error querying the knowledge graph: {str(e)}"

    async def _arun(self, query: str, project_id: str) -> str:
        # For simplicity, we're using the synchronous version in the async context
        # In a production environment, you might want to use an async HTTP client
        return self._run(query, project_id)
