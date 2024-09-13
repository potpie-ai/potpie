import logging
from typing import Any, AsyncGenerator, Dict

from langchain.agents import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    GetCodeGraphFromNodeIdTool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_name_tool import (
    GetCodeGraphFromNodeNameTool,
)

logger = logging.getLogger(__name__)


class NodeIdInput(BaseModel):
    node_id: str = Field(
        ..., description="The ID of the node to retrieve the graph for"
    )


class NodeNameInput(BaseModel):
    node_name: str = Field(
        ..., description="The name of the node to retrieve the graph for"
    )


class CodeGraphRetrievalAgent:
    def __init__(self, llm, sql_db: Session):
        self.llm = llm
        self.sql_db = sql_db
        self.history_manager = ChatHistoryService(sql_db)
        self.repo_id = None
        self.tools = [
            StructuredTool.from_function(
                func=self._run_get_code_graph_from_node_id,
                name="GetCodeGraphFromNodeId",
                description="Get a code graph for a specific node ID",
                args_schema=NodeIdInput,
            ),
            StructuredTool.from_function(
                func=self._run_get_code_graph_from_node_name,
                name="GetCodeGraphFromNodeName",
                description="Get a code graph for a specific node name",
                args_schema=NodeNameInput,
            ),
        ]
        self.agent_executor = None

    def _run_get_code_graph_from_node_id(self, node_id: str) -> Dict[str, Any]:
        tool = GetCodeGraphFromNodeIdTool(self.sql_db)
        return tool.run(repo_id=self.repo_id, node_id=node_id)

    def _run_get_code_graph_from_node_name(self, node_name: str) -> Dict[str, Any]:
        logger.info(
            f"Running get_code_graph_from_node_name with repo_id: {self.repo_id}, node_name: {node_name}"
        )
        tool = GetCodeGraphFromNodeNameTool(self.sql_db)
        return tool.run(repo_id=self.repo_id, node_name=node_name)

    async def _create_agent_executor(self) -> AgentExecutor:
        system_prompt = """You are an AI assistant specialized in retrieving code graphs from a knowledge graph.
Your task is to assist users in finding specific code graphs based on node names or IDs.
Use the GetCodeGraphFromNodeId tool when you have a specific node ID, and use the GetCodeGraphFromNodeName tool when you have a node name or as a fallback.
Return the graph data as a JSON string. Return only code and nothing else."""

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        agent_with_history = (
            RunnablePassthrough.assign(chat_history=lambda x: x.get("chat_history", []))
            | agent
        )

        return AgentExecutor(agent=agent_with_history, tools=self.tools, verbose=True)

    async def run(
        self,
        query: str,
        repo_id: str,
        user_id: str,
        conversation_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            logger.info(f"Running CodeGraphRetrievalAgent for repo_id: {repo_id}")
            self.repo_id = repo_id
            if not self.agent_executor:
                self.agent_executor = await self._create_agent_executor()

            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = []
            for msg in history:
                if isinstance(msg, (str, int, float)):
                    validated_history.append(HumanMessage(content=str(msg)))
                elif isinstance(msg, dict):
                    if msg.get("type") == "human":
                        validated_history.append(
                            HumanMessage(content=msg.get("content", ""))
                        )
                    elif msg.get("type") == "ai":
                        validated_history.append(
                            AIMessage(content=msg.get("content", ""))
                        )
                else:
                    validated_history.append(msg)

            inputs = {
                "input": query,
                "chat_history": validated_history,
            }

            async for chunk in self.agent_executor.astream(inputs):
                if isinstance(chunk, dict):
                    content = chunk.get("output", "")
                else:
                    content = str(chunk)

                if content.strip():
                    self.history_manager.add_message_chunk(
                        conversation_id, content, MessageType.AI_GENERATED
                    )
                    yield content

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(
                f"Error during CodeGraphRetrievalAgent run: {str(e)}", exc_info=True
            )
            yield f"An error occurred: {str(e)}"
