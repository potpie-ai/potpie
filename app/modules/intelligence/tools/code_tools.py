import os
from typing import List

import requests
from langchain.tools import StructuredTool, Tool
from pydantic import BaseModel, Field


class KnowledgeGraphInput(BaseModel):
    query: str = Field(
        description="A natural language question to ask the knowledge graph"
    )
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )


class CodeTools:
    @staticmethod
    def ask_knowledge_graph(query: str, project_id: str) -> str:
        """
        Query the code knowledge graph using natural language questions.
        The knowledge graph contains information from various database tables including:
        - inference: key-value pairs of inferred knowledge about APIs and their constituting functions
        - endpoints: API endpoint paths and identifiers
        - explanation: code explanations for function identifiers
        - pydantic: pydantic class definitions
        """
        data = {"project_id": 2, "query": query}
        headers = {"Content-Type": "application/json"}
        kg_query_url = os.getenv("KNOWLEDGE_GRAPH_URL")
        print("hitting KG")
        response = requests.post(kg_query_url, json=data, headers=headers)
        return response.json()

    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        Get a list of LangChain Tool objects for use in agents.
        """
        return [
            StructuredTool.from_function(
                func=cls.ask_knowledge_graph,
                name="Ask Knowledge Graph",
                description="Query the code knowledge graph with specific directed questions using natural language. Do not use this to query code directly.",
                args_schema=KnowledgeGraphInput,
            ),
        ]
