import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from sqlalchemy.orm import Session
from typing_extensions import TypedDict

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.agents.unit_test_agent import (
    kickoff_unit_test_agent,
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
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)

logger = logging.getLogger(__name__)


class UnitTestAgent:
    def __init__(self, db: Session):
        self.db = db
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)

    @lru_cache(maxsize=2)
    async def _get_prompts(self) -> Dict[PromptType, PromptResponse]:
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            "UNIT_TEST_AGENT", [PromptType.SYSTEM, PromptType.HUMAN]
        )
        return {prompt.type: prompt for prompt in prompts}

    async def _classify_query(
        self, query: str, history: List[HumanMessage], provider_service: ProviderService
    ):
        prompt = ClassificationPrompts.get_classification_prompt(AgentType.UNIT_TEST)
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

    async def _stream_unit_test_agent(self, state: State, writer: StreamWriter):
        async for chunk in self.execute(
            state["query"],
            state["project_id"],
            state["user_id"],
            state["conversation_id"],
            state["node_ids"],
        ):
            writer(chunk)

    def _create_graph(self):
        graph_builder = StateGraph(UnitTestAgent.State)
        graph_builder.add_node("unit_test_agent", self._stream_unit_test_agent)
        graph_builder.add_edge(START, "unit_test_agent")
        graph_builder.add_edge("unit_test_agent", END)
        return graph_builder.compile()

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
    ):
        state = {
            "query": query,
            "project_id": project_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "node_ids": node_ids,
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
    ) -> AsyncGenerator[str, None]:
        try:
            provider_service = ProviderService(self.db, user_id)
            prompts = await self._get_prompts()
            system_prompt = prompts.get(PromptType.SYSTEM)
            human_prompt = prompts.get(PromptType.HUMAN)

            if not system_prompt or not human_prompt:
                raise ValueError("Required prompts not found for UNIT_TEST_AGENT")

            citations = []
            if not node_ids:
                content = "It looks like there is no context selected. Please type @ followed by file or function name to interact with the unit test agent"
                self.history_manager.add_message_chunk(
                    conversation_id,
                    content,
                    MessageType.AI_GENERATED,
                    citations=citations,
                )
                yield json.dumps({"citations": [], "message": content})
                self.history_manager.flush_message_buffer(
                    conversation_id, MessageType.AI_GENERATED
                )
                return

            history = self.history_manager.get_session_history(user_id, conversation_id)
            for node in node_ids:
                history.append(
                    HumanMessage(
                        content=f"{node.name}: {GetCodeFromNodeIdTool(self.db, user_id).run(project_id, node.node_id)}"
                    )
                )
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
                # Get CrewAI LLM once and store it
                crew_ai_llm = provider_service.get_large_llm(
                    agent_type=AgentProvider.CREWAI
                )
                test_response = await kickoff_unit_test_agent(
                    query,
                    validated_history,
                    project_id,
                    node_ids,
                    self.db,
                    crew_ai_llm,
                    user_id,
                )

                if hasattr(test_response, "pydantic"):
                    citations = test_response.pydantic.citations
                    response = test_response.pydantic.response
                else:
                    citations = []
                    response = test_response.get("response", str(test_response))

                tool_results = [
                    SystemMessage(
                        content=f"Unit testing agent response, this is not visible to user:\n {response}"
                    )
                ]

            # Format messages for final response
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
                *[{"role": "system", "content": msg.content} for msg in tool_results],
                {"role": "user", "content": human_prompt.text.format(input=query)},
            ]

            citations = self.agents_service.format_citations(citations)

            try:
                async_generator = await provider_service.call_llm(
                    messages=messages, size="large", stream=True
                )
                async for chunk in async_generator:
                    content = chunk
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
                logger.error(f"Unit test generation failed: {e}")
                yield json.dumps({"error": f"Unit test generation failed: {str(e)}"})

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during UnitTestAgent run: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"
