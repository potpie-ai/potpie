import asyncio
import logging
from typing import AsyncGenerator, Dict, Any

from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_name_tool import GetCodeFromNodeNameTool
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import GetCodeFromNodeIdTool

logger = logging.getLogger(__name__)

class NodeIdInput(BaseModel):
    repo_id: str = Field(..., description="The ID of the repository")
    node_id: str = Field(..., description="The ID of the node to retrieve code from")

class NodeNameInput(BaseModel):
    repo_id: str = Field(..., description="The ID of the repository")
    node_name: str = Field(..., description="The name of the node to retrieve code from")

class CodeRetrievalAgent:
    def __init__(self, openai_key: str, sql_db: Session):
        self.llm = ChatOpenAI(api_key=openai_key, temperature=0.7, model_kwargs={"stream": True})
        self.sql_db = sql_db
        self.history_manager = ChatHistoryService(sql_db)
        self.tools = [
            StructuredTool.from_function(
                func=self._run_get_code_from_node_id,
                name="GetCodeFromNodeId",
                description="Use this tool when you have a specific node ID (a string of letters and numbers) to retrieve code. Input: repo_id and node_id.",
                args_schema=NodeIdInput
            ),
            StructuredTool.from_function(
                func=self._run_get_code_from_node_name,
                name="GetCodeFromNodeName",
                description="Use this tool when you have a node name (usually a word or phrase) to retrieve code. Only use if no node ID is provided. Input: repo_id and node_name.",
                args_schema=NodeNameInput
            )
        ]
        self.agent_executor = None

    def _run_get_code_from_node_id(self, repo_id: str, node_id: str) -> str:
        tool = GetCodeFromNodeIdTool(self.sql_db)
        return tool.run(repo_id=repo_id, node_id=node_id)

    def _run_get_code_from_node_name(self, repo_id: str, node_name: str) -> str:
        tool = GetCodeFromNodeNameTool(self.sql_db)
        return tool.run(repo_id=repo_id, node_name=node_name)

    async def _create_agent_executor(self) -> AgentExecutor:
        system_prompt = """You are an AI assistant specialized in retrieving code from a knowledge graph.
Your task is to assist users in finding specific code snippets based on node IDs or names.
- If a node ID is provided (usually a string of letters and numbers), always use the GetCodeFromNodeId tool.
- Only use the GetCodeFromNodeName tool if no node ID is given and you have a node name (usually a word or phrase).
Use the provided tools to retrieve the code. Return ONLY the code snippet, without any additional explanations or comments."""

        human_prompt = """Please find the code for this query: {input}"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="tool_results"),
            HumanMessage(content=human_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
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

            tool_results = []

            inputs = {
                "input": f"Query: {query}\nRepository ID: {project_id}",
                "chat_history": validated_history,
                "tool_results": tool_results,
                "agent_scratchpad": [],
            }

            logger.debug(f"Inputs to agent: {inputs}")

            async for chunk in self.agent_executor.astream(inputs):
                content = chunk.get('output', '')
                if content.strip():  # Only yield non-empty content
                    self.history_manager.add_message_chunk(
                        conversation_id, content, MessageType.AI_GENERATED
                    )
                    yield content

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during CodeRetrievalAgent run: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"