import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.agents.blast_radius_agent import (
    kickoff_blast_radius_agent,
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
from app.modules.intelligence.provider.provider_service import ProviderService

logger = logging.getLogger(__name__)


class CodeChangesChatAgent:
    def __init__(self, db: Session):
        self.db = db
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.system_message = None
        self.chain = None

    @lru_cache(maxsize=2)
    async def _get_prompts(self) -> Dict[PromptType, PromptResponse]:
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            "CODE_CHANGES_AGENT", [PromptType.SYSTEM, PromptType.HUMAN]
        )
        for prompt in prompts:
            if prompt.type == PromptType.SYSTEM:
                self.system_message = prompt.text
            elif prompt.type == PromptType.HUMAN:
                self.human_message_template = prompt.text

    async def _classify_query(
        self, query: str, history: List[HumanMessage], provider_service: ProviderService
    ):
        if not self.system_message or not self.human_message_template:
            await self._get_prompts()
        prompt = ClassificationPrompts.get_classification_prompt(AgentType.CODE_CHANGES)
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
            response = await provider_service.call_llm(messages=messages, size="small")
            return parser.parse(response).classification
        except Exception:
            logger.warning("Classification failed")
            return ClassificationResult.AGENT_REQUIRED

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        try:
            provider_service = ProviderService(self.db, user_id)

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
                blast_radius_result = await kickoff_blast_radius_agent(
                    query, project_id, node_ids, self.db, user_id
                )

                if blast_radius_result.pydantic:
                    citations = blast_radius_result.pydantic.citations
                    response = blast_radius_result.pydantic.response
                else:
                    citations = []
                    response = blast_radius_result.raw

                tool_results = [
                    SystemMessage(content=f"Blast Radius Agent result: {response}")
                ]

            messages = [
                {"role": "system", "content": self.system_message},
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
                    "content": self.human_message_template.format(input=query),
                },
            ]

            try:
                async for chunk in await provider_service.call_llm(
                    messages=messages, size="small", stream=True
                ):
                    content = chunk
                    self.history_manager.add_message_chunk(
                        conversation_id,
                        content,
                        MessageType.AI_GENERATED,
                        citations=(
                            citations
                            if classification == ClassificationResult.AGENT_REQUIRED
                            else None
                        ),
                    )
                    yield json.dumps(
                        {
                            "citations": (
                                citations
                                if classification == ClassificationResult.AGENT_REQUIRED
                                else []
                            ),
                            "message": content,
                        }
                    )
            except Exception as e:
                logger.warning(f"CodeChangesAgent streaming failed: {e}")

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(
                f"Error during CodeChangesChatAgent run: {str(e)}", exc_info=True
            )
            yield f"An error occurred: {str(e)}"
