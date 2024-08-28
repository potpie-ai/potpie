import logging
from typing import List
from langchain.tools import Tool

from app.modules.github.github_service import GithubService
from app.modules.utils.neo4j_helper import Neo4jGraph

neo4j_graph = Neo4jGraph()

class CodeTools:
    """
    A class that provides code tools for generating code for endpoint identifiers,
    fetching pydantic definitions, and querying the knowledge graph.
    """

    @staticmethod
    def get_flow(endpoint_id: str, project_id: str) -> tuple:
        flow = ()
        nodes_pro = neo4j_graph.find_outbound_neighbors(
            endpoint_id=endpoint_id, project_id=project_id, with_bodies=True
        )
        for node in nodes_pro:
            if "id" in node:
                flow += (node["id"],)
            elif "neighbor" in node:
                flow += (node["neighbor"]["id"],)
        return flow

    @staticmethod
    def get_code_for_function(function_identifier: str) -> str:
        node = neo4j_graph.get_node_by_id(function_identifier, project_id="")
        return GithubService.fetch_method_from_repo(node)

    @staticmethod
    def get_node(function_identifier: str, project_details: dict):
        return neo4j_graph.get_node_by_id(function_identifier, project_details["id"])

    @staticmethod
    def get_node_by_id(node_id: str, project_id: str):
        return neo4j_graph.get_node_by_id(node_id, project_id)

    @staticmethod
    def get_code(identifier: str, project_id: str) -> str:
        """
        Get the code for the specified endpoint identifier.
        """
        code = ""
        nodes = CodeTools.get_flow(identifier, project_id)
        for node_id in nodes:
            node = CodeTools.get_node_by_id(node_id, project_id)
            if node:
                code += f"\n{node['id']}\ncode:\n{CodeTools.get_code_for_function(node['id'])}"
        return code

    @staticmethod
    def get_pydantic_definition(classname: str, project_id: str) -> str:
        """
        Get the pydantic class definition for a given class name.
        """
        inheritance_tree = neo4j_graph.get_class_hierarchy(classname, project_id)
        class_definition_added = ""
        for class_node in inheritance_tree:
            class_content = CodeTools.get_code_for_function(class_node['id'])
            class_definition_added += f"{class_content}\n\n"
        return class_definition_added

    @staticmethod
    def get_pydantic_definitions(classnames: List[str], project_id: str) -> str:
        """
        Get the pydantic class definitions for a list of class names.
        """
        definitions = ""
        try:
            inheritance_nodes = neo4j_graph.get_multiple_class_hierarchies(classnames, project_id)
            for class_node in inheritance_nodes:
                class_content = CodeTools.get_code_for_function(class_node['id'])
                definitions += f"{class_content}\n\n"
        except Exception as e:
            logging.exception(f"project_id: {project_id} something went wrong during fetching definition for {classnames}")
        return definitions

    @staticmethod
    def ask_knowledge_graph(query: str, project_id: str) -> str:
        """
        Query the code knowledge graph using natural language questions.
        """
        try:
            result = neo4j_graph.query_knowledge_graph(query, project_id)
            return str(result)  # Ensure the result is a string
        except Exception as e:
            return f"Error querying the knowledge graph: {str(e)}"

    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        Get a list of LangChain Tool objects for use in agents.
        """
        return [
            Tool(
                name="Get Code",
                func=cls.get_code,
                description="Get accurate code context for given endpoint identifier. Use this to fetch actual code."
            ),
            Tool(
                name="Get Pydantic Definition",
                func=cls.get_pydantic_definition,
                description="Get the pydantic class definition for a single class name."
            ),
            Tool(
                name="Get Pydantic Definitions",
                func=cls.get_pydantic_definitions,
                description="Get the pydantic class definitions for a list of class names."
            ),
            Tool(
                name="Ask Knowledge Graph",
                func=cls.ask_knowledge_graph,
                description="Query the code knowledge graph with specific directed questions using natural language. Do not use this to query code directly."
            )
        ]