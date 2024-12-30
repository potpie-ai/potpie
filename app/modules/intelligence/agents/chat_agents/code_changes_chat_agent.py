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
import asyncio
from app.modules.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class CodeChangesChatAgent:
    def __init__(self, mini_llm, llm, db: Session):
        self.mini_llm = mini_llm
        self._llm = None
        self._llm_provider = ProviderService(db)
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.chain = None
        self.db = db
        self.llm_rate_limiter = RateLimiter(name="LLM_API")
        logger.debug("Rate limiter initialized for CodeChangesChatAgent")

    async def _get_llm(self):
        try:
            await asyncio.wait_for(
                self.llm_rate_limiter.acquire(),
                timeout=30
            )
            logger.debug("Rate limiter acquired for LLM call")
            
            if self._llm is None:
                self._llm = await self._llm_provider.get_small_llm(agent_type=AgentType.LANGCHAIN)
            return self._llm

        except asyncio.TimeoutError:
            logger.error("Timeout waiting for rate limiter")
            raise Exception("Service is currently overloaded. Please try again later.")
        except Exception as e:
            if "429" in str(e) or "quota exceeded" in str(e).lower():
                self.llm_rate_limiter.handle_quota_exceeded()
                logger.error("LLM API quota exceeded")
            logger.error(f"Error getting LLM: {str(e)}", exc_info=True)
            raise

    @lru_cache(maxsize=2)
    async def _get_prompts(self) -> Dict[PromptType, PromptResponse]:
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            "CODE_CHANGES_AGENT", [PromptType.SYSTEM, PromptType.HUMAN]
        )
        return {prompt.type: prompt for prompt in prompts}

    async def _create_chain(self) -> RunnableSequence:
        prompts = await self._get_prompts()
        system_prompt = prompts.get(PromptType.SYSTEM)
        human_prompt = prompts.get(PromptType.HUMAN)

        if not system_prompt or not human_prompt:
            raise ValueError("Required prompts not found for CODE_CHANGES_AGENT")

        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt.text),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="tool_results"),
                HumanMessagePromptTemplate.from_template(human_prompt.text),
            ]
        )
        llm = await self._get_llm()
        return prompt_template | llm

    async def _classify_query(self, query: str, history: List[HumanMessage]):
        prompt = ClassificationPrompts.get_classification_prompt(AgentType.CODE_CHANGES)
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
                blast_radius_result = await kickoff_blast_radius_agent(
                    query,
                    project_id,
                    node_ids,
                    self.db,
                    user_id,
                    self.mini_llm,
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

            inputs = {
                "history": validated_history,
                "tool_results": tool_results,
                "input": query,
            }

            logger.debug(f"Inputs to LLM: {inputs}")

            full_response = ""
            citations = self.agents_service.format_citations(citations)
            async for chunk in self.chain.astream(inputs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += content
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

            logger.debug(f"Full LLM response: {full_response}")
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(
                f"Error during CodeChangesChatAgent run: {str(e)}", exc_info=True
            )
            yield f"An error occurred: {str(e)}"
