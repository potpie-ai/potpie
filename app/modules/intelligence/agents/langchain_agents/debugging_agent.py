import asyncio
import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.crewai_agents.rag_agent import kickoff_rag_crew
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.prompts.prompt_schema import PromptResponse, PromptType
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.tools.kg_based_tools.code_tools import CodeTools

logger = logging.getLogger(__name__)


class DebuggingAgent:
    def __init__(self, llm, db: Session):
        self.llm = llm
        self.history_manager = ChatHistoryService(db)
        self.tools = CodeTools.get_tools()
        self.prompt_service = PromptService(db)
        self.chain = None
        self.db = db

    @lru_cache(maxsize=2)
    async def _get_prompts(self) -> Dict[PromptType, PromptResponse]:
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            "DEBUGGING_AGENT", [PromptType.SYSTEM, PromptType.HUMAN]
        )
        return {prompt.type: prompt for prompt in prompts}

    async def _create_chain(self) -> RunnableSequence:
        prompts = await self._get_prompts()
        system_prompt = prompts.get(PromptType.SYSTEM)
        human_prompt = prompts.get(PromptType.HUMAN)

        if not system_prompt or not human_prompt:
            raise ValueError("Required prompts not found for DEBUGGING_AGENT")

        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt.text),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="tool_results"),
                HumanMessagePromptTemplate.from_template(human_prompt.text),
            ]
        )
        return prompt_template | self.llm

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
        logs: str = "",
        stacktrace: str = "",
    ) -> AsyncGenerator[str, None]:
        try:
            if not self.chain:
                self.chain = await self._create_chain()

            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            rag_result = await kickoff_rag_crew(
                query,
                project_id,
                [msg.content for msg in validated_history if isinstance(msg, HumanMessage)],
                node_ids,
                self.db
            )
            
            tool_results = [SystemMessage(content=f"RAG Agent result: {[node.model_dump() for node in rag_result.pydantic.response]}")]

            full_query = f"Query: {query}\nProject ID: {project_id}\nLogs: {logs}\nStacktrace: {stacktrace}"
            inputs = {
                "history": validated_history,
                "tool_results": tool_results,
                "input": full_query,
            }

            logger.debug(f"Inputs to LLM: {inputs}")

            full_response = ""
            async for chunk in self.chain.astream(inputs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += content
                self.history_manager.add_message_chunk(
                    conversation_id, content, MessageType.AI_GENERATED
                )
                yield json.dumps({"citations": rag_result.pydantic.citations, "message":full_response})

            logger.debug(f"Full LLM response: {full_response}")

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during DebuggingAgent run: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"
