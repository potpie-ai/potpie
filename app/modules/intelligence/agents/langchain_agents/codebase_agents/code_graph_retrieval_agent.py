import json
import logging
from typing import Any, AsyncGenerator, Dict, List

from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.kg_based_tools.get_code_graph_from_node_id_tool import GetCodeGraphFromNodeIdTool
from app.modules.intelligence.tools.kg_based_tools.get_code_graph_from_node_name_tool import GetCodeGraphFromNodeNameTool

logger = logging.getLogger(__name__)

class NodeIdInput(BaseModel):
    repo_id: str = Field(..., description="The ID of the repository")
    node_id: str = Field(..., description="The ID of the node to retrieve the graph for")

class NodeNameInput(BaseModel):
    repo_id: str = Field(..., description="The ID of the repository")
    node_name: str = Field(..., description="The name of the node to retrieve the graph for")

class CodeGraphRetrievalAgent:
    def __init__(self, llm: BaseChatModel, sql_db: Session):
        self.llm = llm
        self.sql_db = sql_db
        self.history_manager = ChatHistoryService(sql_db)
        self.tools = [
            StructuredTool.from_function(
                func=self._run_get_code_graph_from_node_id,
                name="GetCodeGraphFromNodeId",
                description="Use this tool when you have a specific node ID to retrieve the code graph.",
                args_schema=NodeIdInput,
            ),
            StructuredTool.from_function(
                func=self._run_get_code_graph_from_node_name,
                name="GetCodeGraphFromNodeName",
                description="Use this tool when you have a node name to retrieve the code graph, or as a fallback if GetCodeGraphFromNodeId fails.",
                args_schema=NodeNameInput,
            ),
        ]
        self.agent_executor = None

    def _run_get_code_graph_from_node_id(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        tool = GetCodeGraphFromNodeIdTool(self.sql_db)
        result = tool.run(repo_id=repo_id, node_id=node_id)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def _run_get_code_graph_from_node_name(self, repo_id: str, node_name: str) -> Dict[str, Any]:
        tool = GetCodeGraphFromNodeNameTool(self.sql_db)
        result = tool.run(repo_id=repo_id, node_name=node_name)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    async def _create_agent_executor(self) -> AgentExecutor:
        system_prompt = """You are an AI assistant specialized in retrieving code graphs from a knowledge graph.
Your task is to assist users in finding specific code graphs based on node IDs or names.
- If GetCodeGraphFromNodeId fails, use GetCodeGraphFromNodeName as a fallback.
- When using GetCodeGraphFromNodeName, if it initially fails, consider using the full query as the node name.
- If both tools fail, inform the user that the requested information could not be found.
Return ONLY the JSON representation of the code graph, without any additional explanations or comments."""

        human_prompt = """Please find the code graph for this query: {input}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                HumanMessage(content=human_prompt),
            ]
        )

        agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            if not self.agent_executor:
                self.agent_executor = await self._create_agent_executor()

            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (HumanMessage(content=str(msg)) if isinstance(msg, (str, int, float)) else msg)
                for msg in history
            ]

            inputs = {
                "input": f"Query: {query}\nRepository ID: {project_id}",
                "chat_history": validated_history,
            }

            logger.debug(f"Inputs to agent: {inputs}")

            result = await self.agent_executor.ainvoke(inputs)

            # Extract the agent's output
            agent_output = result.get("output", "")

            # Ensure the result is a valid JSON string
            try:
                json_result = json.loads(agent_output)
                output = json.dumps(json_result, indent=2)
            except json.JSONDecodeError:
                if "error" in agent_output:
                    output = json.dumps({"error": agent_output})
                else:
                    output = json.dumps({"error": "Invalid JSON response from agent", "raw_output": agent_output})

            self.history_manager.add_message_chunk(conversation_id, output, MessageType.AI_GENERATED)
            yield output

            self.history_manager.flush_message_buffer(conversation_id, MessageType.AI_GENERATED)

        except Exception as e:
            logger.error(f"Error during CodeGraphRetrievalAgent run: {str(e)}", exc_info=True)
            error_output = json.dumps({"error": f"An error occurred: {str(e)}"})
            yield error_output
