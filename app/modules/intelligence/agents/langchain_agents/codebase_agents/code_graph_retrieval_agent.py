from typing import Any, AsyncGenerator, Dict

from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.get_code_flow_from_node_id_tool import (
    GetCodeFlowFromNodeIdTool,
)


class CodeGraphRetrievalAgent:
    def __init__(self, sql_db: Session):
        self.sql_db = sql_db
        self.history_manager = ChatHistoryService(sql_db)
        self.code_flow_tool = GetCodeFlowFromNodeIdTool(sql_db)

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            print(
                f"CodeGraphRetrievalAgent.run called with query: {query}, project_id: {project_id}"
            )

            # Extract node_id from query (implement this method based on your query format)
            node_id = self._extract_node_id(query)

            if not node_id:
                yield "To generate a code flow, please provide a specific node ID."
                return

            result = self.code_flow_tool.run(project_id, node_id)

            if isinstance(result, dict) and "code_flow" in result:
                code_flow = result["code_flow"]
                output = self._format_code_flow(code_flow)
                yield output

                self.history_manager.add_message_chunk(
                    conversation_id, output, MessageType.AI_GENERATED
                )
                self.history_manager.flush_message_buffer(
                    conversation_id, MessageType.AI_GENERATED
                )
            else:
                error_message = result.get("error", "Unknown error occurred")
                print(f"Error in code flow result: {error_message}")
                yield f"Error retrieving code flow: {error_message}"

        except Exception as e:
            print(f"Error during CodeGraphRetrievalAgent run: {str(e)}")
            yield f"An error occurred: {str(e)}"

    def _extract_node_id(self, query: str) -> str:
        # Implement node ID extraction logic here
        # This is a placeholder and should be implemented based on your specific requirements
        return ""

    def _format_code_flow(self, code_flow: Dict[str, Any]) -> str:
        nodes = code_flow.get("nodes", [])
        relationships = code_flow.get("relationships", [])

        output = "Code Flow:\n\n"
        output += "Nodes:\n"
        for node in nodes:
            output += f"- ID: {node['id']}\n"
            output += f"  Name: {node['name']}\n"
            output += f"  Type: {node['type']}\n"
            output += f"  File: {node['file_path']}\n"
            output += f"  Lines: {node['start_line']} - {node['end_line']}\n\n"

        output += "Relationships:\n"
        for rel in relationships:
            output += f"- {rel['source']} --[{rel['type']}]--> {rel['target']}\n"

        return output
