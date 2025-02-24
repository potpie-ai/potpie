import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, Dict, List, TypedDict

from langchain.schema import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
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
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentProvider,
)

logger = logging.getLogger(__name__)


class DebuggingChatAgent:
    def __init__(self, db: Session):
        self.db = db
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.chain = None

    @lru_cache(maxsize=2)
    async def _get_prompts(self) -> Dict[PromptType, PromptResponse]:
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            "DEBUGGING_AGENT", [PromptType.SYSTEM, PromptType.HUMAN]
        )
        return {prompt.type: prompt for prompt in prompts}

    async def _classify_query(
        self, query: str, history: List[HumanMessage], provider_service: ProviderService
    ):
        prompt = ClassificationPrompts.get_classification_prompt(AgentType.DEBUGGING)
        inputs = {"query": query, "history": [msg.content for msg in history[-5:]]}

        parser = PydanticOutputParser(pydantic_object=ClassificationResponse)
        format_instructions = parser.get_format_instructions()

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Query: {inputs['query']}\nHistory: {inputs['history']}\n\n{format_instructions}",
            },
        ]

        try:
            result = await provider_service.call_llm_with_structured_output(
                messages=messages, output_schema=ClassificationResponse, size="large"
            )
            return result.classification
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult.AGENT_REQUIRED

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
            if isinstance(chunk, str):
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
        try:
            provider_service = ProviderService(self.db, user_id)
            prompts = await self._get_prompts()
            system_prompt = prompts.get(PromptType.SYSTEM)
            human_prompt = prompts.get(PromptType.HUMAN)

            if not system_prompt or not human_prompt:
                raise ValueError("Required prompts not found for DEBUGGING_AGENT")

            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            classification = await self._classify_query(
                query, validated_history, provider_service
            )

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
                    provider_service.get_large_llm(agent_type=AgentProvider.CREWAI),
                    provider_service.get_small_llm(agent_type=AgentProvider.CREWAI),
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
            else:
                full_query = f"Query: {query}\nProject ID: {project_id}\nLogs: {logs}\nStacktrace: {stacktrace}"

                messages = [
                    {"role": "system", "content": system_prompt.text},
                    *[
                        {
                            "role": (
                                "user" if isinstance(msg, HumanMessage) else "assistant"
                            ),
                            "content": msg.content,
                        }
                        for msg in validated_history
                    ],
                    *[
                        {"role": "system", "content": result.content}
                        for result in tool_results
                    ],
                    {
                        "role": "user",
                        "content": human_prompt.text.format(input=full_query),
                    },
                ]

                try:
                    response = await provider_service.call_llm(
                        messages=messages, size="large"
                    )
                    content = response
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
                except Exception as e:
                    logger.error(f"Debugging generation failed: {e}")
                    yield json.dumps(
                        {"error": f"Debugging generation failed: {str(e)}"}
                    )

                self.history_manager.flush_message_buffer(
                    conversation_id, MessageType.AI_GENERATED
                )

        except Exception as e:
            logger.error(
                f"Error during DebuggingChatAgent run: {str(e)}", exc_info=True
            )
            yield f"An error occurred: {str(e)}"
