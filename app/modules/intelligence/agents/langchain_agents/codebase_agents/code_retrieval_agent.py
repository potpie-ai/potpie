import logging
import re
from typing import Any, AsyncGenerator, Dict, Tuple

from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.code_query_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_from_node_name_tool import (
    GetCodeFromNodeNameTool,
)

logger = logging.getLogger(__name__)


class NodeIdInput(BaseModel):
    repo_id: str = Field(..., description="The ID of the repository")
    node_id: str = Field(..., description="The ID of the node to retrieve code from")


class NodeNameInput(BaseModel):
    repo_id: str = Field(..., description="The ID of the repository")
    node_name: str = Field(
        ..., description="The name of the node to retrieve code from"
    )


class CodeRetrievalAgent:
    def __init__(self, llm, sql_db: Session):
        self.llm = llm
        self.sql_db = sql_db
        self.history_manager = ChatHistoryService(sql_db)
        self.tools = [
            StructuredTool.from_function(
                func=self._run_get_code_from_node_id,
                name="GetCodeFromNodeId",
                description="Use this tool when you have a specific node ID to retrieve code.",
                args_schema=NodeIdInput,
            ),
            StructuredTool.from_function(
                func=self._run_get_code_from_node_name,
                name="GetCodeFromNodeName",
                description="Use this tool when you have a node name to retrieve code, or as a fallback if GetCodeFromNodeId fails.",
                args_schema=NodeNameInput,
            ),
        ]
        self.agent_executor = None

    def _run_get_code_from_node_id(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        tool = GetCodeFromNodeIdTool(self.sql_db)
        return tool.run(repo_id=repo_id, node_id=node_id)

    def _run_get_code_from_node_name(
        self, repo_id: str, node_name: str
    ) -> Dict[str, Any]:
        tool = GetCodeFromNodeNameTool(self.sql_db)
        return tool.run(repo_id=repo_id, node_name=node_name)

    def _extract_node_id(self, query: str) -> Tuple[str, str]:
        node_id_pattern = r"\b[a-f0-9]{32}\b"
        match = re.search(node_id_pattern, query)
        if match:
            node_id = match.group(0)
            remaining_query = query.replace(node_id, "").strip()
            return node_id, remaining_query
        return "", query

    async def _create_agent_executor(self) -> AgentExecutor:
        system_prompt = """You are an AI assistant specialized in retrieving code from a knowledge graph.
Your task is to assist users in finding specific code snippets based on node IDs or names.
- If GetCodeFromNodeId fails, use GetCodeFromNodeName as a fallback.
- When using GetCodeFromNodeName, if it initally fails, consider using the full query as the node name.
Return ONLY the code snippet, without any additional explanations or comments or any other text."""

        human_prompt = """Please find the code for this query: {input}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                HumanMessage(content=human_prompt),
            ]
        )

        agent = self.llm  # Use the provided LLM directly

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            prompt=prompt,  # Use the prompt here
            verbose=True,
        )

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
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            node_id, remaining_query = self._extract_node_id(query)

            if node_id:
                result = self._run_get_code_from_node_id(project_id, node_id)
                if not isinstance(result, dict) or "error" not in result:
                    if isinstance(result, dict) and "code_content" in result:
                        yield result["code_content"]
                    elif isinstance(result, str):
                        yield result
                    return

            inputs = {
                "input": f"Query: {query}\nRepository ID: {project_id}",
                "chat_history": validated_history,
            }

            logger.debug(f"Inputs to agent: {inputs}")

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
                f"Error during CodeRetrievalAgent run: {str(e)}", exc_info=True
            )
            yield f"An error occurred: {str(e)}"
