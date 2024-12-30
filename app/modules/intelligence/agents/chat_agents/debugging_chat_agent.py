import json
import logging
import time
import asyncio
from functools import lru_cache
from typing import AsyncGenerator, Dict, List, TypedDict

from langchain.schema import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.agents.debug_rag_agent import (
    kickoff_debug_rag_agent,
)
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.prompts.classification_prompts import (
    AgentType,
    ClassificationPrompts,
    ClassificationResponse,
    ClassificationResult,
)
from app.modules.intelligence.prompts.prompt_schema import PromptResponse, PromptType
from app.modules.intelligence.prompts.prompt_service import PromptService

from app.modules.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class DebuggingChatAgent:
    def __init__(self, mini_llm, reasoning_llm, db: Session):
        self.mini_llm = mini_llm
        self.llm = reasoning_llm
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.chain = None
        self.db = db
        self.llm_rate_limiter = RateLimiter(name="LLM_API")
        logger.debug("Rate limiter initialized for DebuggingChatAgent")

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
        try:
            await asyncio.wait_for(
                self.llm_rate_limiter.acquire(),
                timeout=30
            )
            logger.debug("Rate limiter acquired for classification query")

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

        except asyncio.TimeoutError:
            logger.error("Timeout waiting for rate limiter")
            raise Exception("Service is currently overloaded. Please try again later.")
        except Exception as e:
            if "429" in str(e) or "quota exceeded" in str(e).lower():
                self.llm_rate_limiter.handle_quota_exceeded()
                logger.error("LLM API quota exceeded")
            logger.error(f"Error in classification query: {str(e)}", exc_info=True)
            raise

    class State(TypedDict):
        query: str
        project_id: str
        user_id: str
        conversation_id: str
        node_ids: List[NodeContext]
        logs: str
        stacktrace: str

    async def _stream_rag_agent(self, state: State, writer: StreamWriter):
        async for chunk in self.execute(
            state["query"],
            state["project_id"],
            state["user_id"],
            state["conversation_id"],
            state["node_ids"],
            state["logs"],
            state["stacktrace"],
        ):
            writer(chunk)

    def _create_graph(self):
        graph_builder = StateGraph(DebuggingChatAgent.State)
        graph_builder.add_node("rag_agent", self._stream_rag_agent)
        graph_builder.add_edge(START, "rag_agent")
        graph_builder.add_edge("rag_agent", END)
        return graph_builder.compile()

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
        logs: str = "",
        stacktrace: str = "",
    ):
        state = {
            "query": query,
            "project_id": project_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "node_ids": node_ids,
            "logs": logs,
            "stacktrace": stacktrace,
        }
        graph = self._create_graph()
        async for chunk in graph.astream(state, stream_mode="custom"):
            yield chunk

    async def execute(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
        logs: str = "",
        stacktrace: str = "",
    ) -> AsyncGenerator[str, None]:
        start_time = time.time()  # Start the timer

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
                async for chunk in kickoff_debug_rag_agent(
                    query,
                    project_id,
                    [
                        msg.content
                        for msg in validated_history
                        if isinstance(msg, HumanMessage)
                    ],
                    node_ids,
                    self.db,
                    self.llm,
                    self.mini_llm,
                    user_id,
                ):
                    content = str(chunk)
                    self.history_manager.add_message_chunk(
                        conversation_id,
                        content,
                        MessageType.AI_GENERATED,
                        citations=citations,
                    )
                    yield json.dumps(
                        {
                            "citations": citations,
                            "message": content,
                        }
                    )

                self.history_manager.flush_message_buffer(
                    conversation_id, MessageType.AI_GENERATED
                )

            if classification != ClassificationResult.AGENT_REQUIRED:
                full_query = f"Query: {query}\nProject ID: {project_id}\nLogs: {logs}\nStacktrace: {stacktrace}"
                inputs = {
                    "history": validated_history,
                    "tool_results": tool_results,
                    "input": full_query,
                }

                logger.debug(f"Inputs to LLM: {inputs}")
                citations = self.agents_service.format_citations(citations)
                full_response = ""
                async for chunk in self.chain.astream(inputs):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_response += content
                    self.history_manager.add_message_chunk(
                        conversation_id,
                        content,
                        MessageType.AI_GENERATED,
                        citations=citations,
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
            logger.error(
                f"Error during DebuggingChatAgent run: {str(e)}", exc_info=True
            )
            yield f"An error occurred: {str(e)}"
