import json
import logging
from typing import AsyncGenerator, List

from langchain.schema import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from sqlalchemy.orm import Session
from typing_extensions import TypedDict

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.agents.code_gen_agent import (
    kickoff_code_generation_crew,
)
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService

logger = logging.getLogger(__name__)


class CodeGenerationChatAgent:
    def __init__(self, mini_llm, llm, db: Session):
        self.mini_llm = mini_llm
        self._llm = None
        self._llm_provider = ProviderService(db)
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.chain = None
        self.db = db

    async def _get_llm(self):
        if self._llm is None:
            self._llm = await self._llm_provider.get_small_llm(agent_type=AgentType.LANGCHAIN)
        return self._llm

    class State(TypedDict):
        query: str
        project_id: str
        user_id: str
        conversation_id: str
        node_ids: List[NodeContext]

    async def _stream_code_gen_agent(self, state: State, writer: StreamWriter):
        async for chunk in self.execute(
            state["query"],
            state["project_id"],
            state["user_id"],
            state["conversation_id"],
            state["node_ids"],
        ):
            writer(chunk)

    def _create_graph(self):
        graph_builder = StateGraph(CodeGenerationChatAgent.State)
        graph_builder.add_node("code_gen_agent", self._stream_code_gen_agent)
        graph_builder.add_edge(START, "code_gen_agent")
        graph_builder.add_edge("code_gen_agent", END)
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
        llm = await self._get_llm()
        async for chunk in graph.astream(state, stream_mode="custom"):
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
            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            citations = []
            async for chunk in kickoff_code_generation_crew(
                query,
                project_id,
                validated_history[-5:],
                node_ids,
                self.db,
                await self._get_llm(),
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

        except Exception as e:
            logger.error(f"Error in code generation: {str(e)}")
            yield json.dumps(
                {"error": f"An error occurred during code generation: {str(e)}"}
            )
