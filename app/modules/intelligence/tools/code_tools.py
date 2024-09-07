import os
from typing import List

import httpx
import requests
from langchain.tools import StructuredTool, Tool
from pydantic import BaseModel, Field

from app.modules.parsing.knowledge_graph.inference_schema import QueryRequest, QueryResponse


class KnowledgeGraphInput(BaseModel):
    query: str = Field(
        description="A natural language question to ask the knowledge graph"
    )
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )
    node_ids: List[str] = Field(
        description="A list of node IDs to narrow down the search context, in case there are none, pass an empty list"
    )


class CodeTools:


    @staticmethod
    def ask_knowledge_graph(query: str, project_id: str = None, node_ids: List[str] = []) -> List[QueryResponse]:
        """
        Query the code knowledge graph using natural language questions.
        The knowledge graph performs a vector search over a corpus of code converted to docstrings at different granularity levels across the codebase, including function, class, and file levels.
        Use this to answer questions about the codebase.
        Args:
            query (str): A natural language question to ask the knowledge graph.
            project_id (str, optional): The project id metadata for the project being evaluated.
            node_ids (List[str], optional): A list of node IDs to narrow down the search context.

        Returns:
            List[QueryResponse]: A list of query responses containing relevant code snippets and their metadata.
        """
        data = QueryRequest(project_id=project_id, query=query, node_ids=node_ids).model_dump()
        headers = {"Content-Type": "application/json", 
                   "Authorization": f"Bearer {os.getenv('INTERNAL_CALL_SECRET')}"}
        kg_query_url = os.getenv("KNOWLEDGE_GRAPH_URL") + "api/v1/query"
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(kg_query_url, json=data, headers=headers)
        response = requests.post(kg_query_url, json=data, headers=headers)
        return [QueryResponse(**item) for item in response.json()]

    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        Get a list of LangChain Tool objects for use in agents.
        """
        return [
            StructuredTool.from_function(
                func=cls.ask_knowledge_graph,
                name="Ask Knowledge Graph",
                description="Query the code knowledge graph with specific directed questions about the codebase using natural language. Do not use this to query code directly.",
                args_schema=KnowledgeGraphInput,
            ),
        ]
