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
        The knowledge graph contains information about every function, class, and file in the codebase.
        Ask it questions about the codebase.
        Inputs for the ask_knowledge_graph method:
        - query (str): A natural language question that the user wants to ask the knowledge graph. This should be a clear and concise question related to the codebase, such as "What functions are defined in the project?" or "How do I use the XYZ class?".

        - project_id (str): The ID of the project being evaluated, this is a UUID.
        """
        data = {"project_id": project_id, "query": query}
        headers = {"Content-Type": "application/json"}
        kg_query_url = os.getenv("KNOWLEDGE_GRAPH_URL")
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
                description=""""
        Query the code knowledge graph using natural language questions.
        The knowledge graph contains information about every function, class, and file in the codebase.
        Ask it questions about the codebase.
        Inputs for the ask_knowledge_graph method:
        - query (str): A natural language question that the user wants to ask the knowledge graph. This should be a clear and concise question related to the codebase, such as "What functions are defined in the project?" or "How do I use the XYZ class?".
        - project_id (str): The ID of the project being evaluated, this is a UUID.
        Do not use this to query code directly.""",
                args_schema=KnowledgeGraphInput,
            ),
        ]
