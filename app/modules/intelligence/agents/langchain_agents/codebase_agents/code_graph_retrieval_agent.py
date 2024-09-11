import logging
from typing import Any, AsyncGenerator, Dict

from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
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
        self.agent_executor = None

    def _run_get_code_graph_from_node_id(self, node_id: str) -> Dict[str, Any]:
        tool = GetCodeGraphFromNodeIdTool(self.sql_db)
        return tool.run(repo_id=self.repo_id, node_id=node_id)

    def _run_get_code_graph_from_node_name(self, node_name: str) -> Dict[str, Any]:
        print(
            f"Running get_code_graph_from_node_name with repo_id: {self.repo_id}, node_name: {node_name}"
        )
        tool = GetCodeGraphFromNodeNameTool(self.sql_db)
        return tool.run(repo_id=self.repo_id, node_name=node_name)

    async def _create_agent_executor(self) -> AgentExecutor:
        system_prompt = """You are an AI assistant specialized in retrieving code graphs from a knowledge graph.
Your task is to assist users in finding specific code graphs based on node names or IDs.
Use the GetCodeGraphFromNodeId tool when you have a specific node ID, and use the GetCodeGraphFromNodeName tool when you have a node name or as a fallback.
Return the graph data as a JSON string.Return only code and nothing else."""

        human_prompt = """Please find the code graph for this query: {input}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content=human_prompt),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = self.llm  # Use the provided LLM directly

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            prompt=prompt,  # Use the prompt here
            verbose=True,
            handle_parsing_errors=True,
        )

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
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            inputs = {
                "input": query,
                "chat_history": validated_history,
            }

            async for chunk in self.agent_executor.astream(inputs):
                content = chunk.get("output", "")
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
