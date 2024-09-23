import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
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
from app.modules.intelligence.agents.agentic_tools.rag_agent import kickoff_rag_crew
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.prompts.classification_prompts import (
    AgentType,
    ClassificationPrompts,
    ClassificationResponse,
    ClassificationResult,
)
from app.modules.intelligence.prompts.prompt_schema import PromptResponse, PromptType
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.tools.kg_based_tools.graph_tools import CodeTools

logger = logging.getLogger(__name__)


class DebuggingAgent:
    def __init__(self, mini_llm, reasoning_llm, db: Session):
        self.mini_llm = mini_llm
        self.llm = reasoning_llm
        self.history_manager = ChatHistoryService(db)
        self.tools = CodeTools.get_kg_tools()
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
        return prompt_template | self.mini_llm

    async def _classify_query(self, query: str, history: List[HumanMessage]):
        prompt = ClassificationPrompts.get_classification_prompt(AgentType.DEBUGGING)
        inputs = {"query": query, "history": [msg.content for msg in history[-5:]]}

        parser = PydanticOutputParser(pydantic_object=ClassificationResponse)
        prompt_with_parser = ChatPromptTemplate.from_template(
            template=prompt,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt_with_parser | self.llm | parser
        response = await chain.ainvoke(input=inputs)

        return response.classification

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

            classification = await self._classify_query(query, validated_history)

            tool_results = []
            citations = []
            if classification == ClassificationResult.AGENT_REQUIRED:
                rag_result = await kickoff_rag_crew(
                    query,
                    project_id,
                    [
                        msg.content
                        for msg in validated_history
                        if isinstance(msg, HumanMessage)
                    ],
                    node_ids,
                    self.db,
                    self.mini_llm,
                )
                if rag_result.pydantic:
                    response = rag_result.pydantic.response
                    citations = rag_result.pydantic.citations
                    result = [node.model_dump() for node in response]
                else:
                    result = rag_result.raw
                    citations = []

                tool_results = [SystemMessage(content=f"RAG Agent result: {result}")]

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
                yield json.dumps(
                    {
                        "citations": citations,
                        "message": content,
                    }
                )

            logger.debug(f"Full LLM response: {full_response}")

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during DebuggingAgent run: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"
