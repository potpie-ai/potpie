from typing import Any, AsyncGenerator, Coroutine, List
from pydantic import BaseModel
from enum import Enum

from app.modules.intelligence.agents_copy.chat_agent import ChatAgentResponse
from ..chat_agent import ChatAgent
from .simple_llm_agent import SimpleLLMAgent
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentType,
)
from app.modules.intelligence.prompts.classification_prompts import (
    ClassificationPrompts,
    AgentType,
)
from app.modules.intelligence.prompts.prompt_service import PromptService, PromptType
import json
import logging

logger = logging.getLogger(__name__)


class AdaptiveAgent(ChatAgent):
    """AdaptiveAgent automatically switches between simple LLM and
    Full RAG agent depending on the query complexity"""

    def __init__(
        self,
        llm_provider: ProviderService,
        prompt_provider: PromptService,
        rag_agent: ChatAgent,
        agent_type: AgentType,
    ):
        classification_prompt = ClassificationPrompts.get_classification_prompt(
            agent_type
        )
        self.llm_provider = llm_provider
        self.classifier = SimpleLLMAgent(
            llm_provider,
            prompt_template=classification_prompt
            + " just return the single classification in response ",
        )
        self.prompt_provider = prompt_provider
        self.agent_type = agent_type

        self.rag_agent = rag_agent

    async def _create_llm_agent(self) -> ChatAgent:
        llm_prompts = await self.prompt_provider.get_prompts_by_agent_id_and_types(
            str(self.agent_type), [PromptType.SYSTEM, PromptType.HUMAN]
        )
        prompts = {prompt.type: prompt for prompt in llm_prompts}
        system_prompt = prompts.get(PromptType.SYSTEM)
        if system_prompt == None:
            # raise ValueError(f"System Prompt for {self.agent_type} not found!!")
            logger.error(f"System Prompt for {self.agent_type} not found!!")

        return SimpleLLMAgent(
            self.llm_provider,
            prompt_template=(
                system_prompt.text
                if system_prompt
                else f"you are a {self.agent_type} agent "
                + " who has complete understading of repo. With the given history of chat: {history} \nAnswer the following query with given info: {query}"
            ),
        )

    async def run(
        self, query: str, history: List[str], node_ids: List[str] | None = None
    ) -> ChatAgentResponse:
        # classify the query into agent needed or not
        classification_response = await self.classifier.run(query, history, node_ids)
        classification = "AGENT_REQUIRED"
        print("Classification response:", classification_response.response)
        try:
            classification_json = json.loads(classification_response.response)
            if (
                classification_json
                and classification_json["classification"] == "LLM_SUFFICIENT"
            ):
                classification = "LLM_SUFFICIENT"
        except Exception:
            pass

        if classification == "AGENT_REQUIRED":
            rag_agent_response = await self.rag_agent.run(query, history, node_ids)
            query += f" with information: {rag_agent_response.response} and citations: {rag_agent_response.citations}"

        # build llm response
        llm_agent = await self._create_llm_agent()
        return await llm_agent.run(query, history, node_ids)

    async def run_stream(
        self, query: str, history: List[str], node_ids: List[str] | None = None
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        classification_response = await self.classifier.run(query, history, node_ids)
        classification = "AGENT_REQUIRED"
        print("Classification response:", classification_response.response)
        try:
            classification_json = json.loads(classification_response.response)
            if (
                classification_json
                and classification_json["classification"] == "LLM_SUFFICIENT"
            ):
                classification = "LLM_SUFFICIENT"
        except Exception:
            pass

        if classification == "AGENT_REQUIRED":
            rag_agent_response = await self.rag_agent.run(query, history, node_ids)
            query += f" with information: {rag_agent_response.response} and citations: {rag_agent_response.citations}"

        # build llm response
        llm_agent = await self._create_llm_agent()
        return await llm_agent.run_stream(query, history, node_ids)
