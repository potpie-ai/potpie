import logging
from typing import Any, AsyncGenerator, Dict, Tuple

from langchain.schema import HumanMessage, SystemMessage
from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_name_tool import (
    GetCodeFromNodeNameTool,
)

logger = logging.getLogger(__name__)


class CodeGraphRetrievalAgent:
    def __init__(self, llm, db: Session):
        self.llm = llm
        self.history_manager = ChatHistoryService(db)
        self.get_code_from_node_id_tool = GetCodeFromNodeIdTool(db)
        self.get_code_from_node_name_tool = GetCodeFromNodeNameTool(db)
        self.neo4j_driver = self._create_neo4j_driver()

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        neo4j_config = config_provider.get_neo4j_config()
        return GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            logger.debug(
                f"CodeGraphRetrievalAgent.run called with query: {query}, project_id: {project_id}"
            )

            query_type, extracted_value = await self._interpret_query(query)

            if query_type == "node_id":
                result = self.get_code_from_node_id_tool.run(
                    project_id, extracted_value
                )
            elif query_type == "node_name":
                result = self.get_code_from_node_name_tool.run(
                    project_id, extracted_value
                )
            else:
                yield "Unable to interpret the query. Please provide a more specific query with a node ID or name."
                return

            if "error" in result:
                yield f"Error retrieving node data: {result['error']}"
                return

            graph_data = self._get_graph_data(project_id, result["node_id"])
            output = self._format_graph_data(graph_data)

            full_response = ""
            for chunk in output.split():  # Simulating streaming for consistency
                full_response += chunk + " "
                self.history_manager.add_message_chunk(
                    conversation_id, chunk + " ", MessageType.AI_GENERATED
                )
                yield chunk + " "

            logger.debug(f"Full response: {full_response}")

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(
                f"Error during CodeGraphRetrievalAgent run: {str(e)}", exc_info=True
            )
            yield f"An error occurred: {str(e)}"

    async def _interpret_query(self, query: str) -> Tuple[str, str]:
        system_message = SystemMessage(
            content="""
        You are an AI assistant that interprets queries about code nodes. Your task is to determine if a query is asking for a node by ID or by name, and extract the relevant information.
        """
        )
        human_message = HumanMessage(
            content=f"""
        Given the following query, determine if it's asking for a node by ID or by name.
        If it's asking for a node by ID, extract the ID. If it's asking for a node by name, extract the name.

        Query: {query}

        Respond in the following format:
        Type: [node_id/node_name]
        Value: [extracted ID or name]

        If you can't determine the type or extract a value, respond with:
        Type: unknown
        Value: none
        """
        )

        messages = [system_message, human_message]
        response = await self.llm.agenerate([messages])
        content = response.generations[0][0].text

        lines = content.strip().split("\n")
        query_type = lines[0].split(": ")[1]
        value = lines[1].split(": ")[1]

        return query_type, value if value != "none" else None

    def _get_graph_data(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        query = """
        MATCH (n:NODE {node_id: $node_id, repoId: $repo_id})
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN n AS node, collect(DISTINCT {type: type(r), direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END, node: related}) AS connections
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, repo_id=repo_id)
            record = result.single()
            return {
                "node": dict(record["node"]),
                "connections": [dict(conn) for conn in record["connections"]],
            }

    def _format_graph_data(self, graph_data: Dict[str, Any]) -> str:
        node = graph_data["node"]
        connections = graph_data["connections"]

        output = "Graph Data:\n\n"
        output += "Central Node:\n"
        output += f"- ID: {node['node_id']}\n"
        output += f"- Name: {node['name']}\n"
        output += f"- Type: {node['type']}\n"
        output += f"- File: {node['file_path']}\n"
        output += f"- Lines: {node['start_line']} - {node['end_line']}\n\n"

        output += "Connections:\n"
        for conn in connections:
            direction = "→" if conn["direction"] == "outgoing" else "←"
            output += f"- {direction} [{conn['type']}] {conn['node']['name']} (ID: {conn['node']['node_id']})\n"

        return output

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()
