from typing import AsyncGenerator

from app.modules.intelligence.agents_copy.chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
)
from .llm_chat import LLM
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
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
        self.classifier = LLM(
            llm_provider,
            prompt_template=classification_prompt
            + "\n the classification output json should be returned as response field in pydantic",
        )
        self.prompt_provider = prompt_provider
        self.agent_type = agent_type

        self.rag_agent = rag_agent

    async def _create_llm(self) -> ChatAgent:
        llm_prompts = await self.prompt_provider.get_prompts_by_agent_id_and_types(
            self.agent_type.value, [PromptType.SYSTEM, PromptType.HUMAN]
        )
        prompts = {prompt.type: prompt for prompt in llm_prompts}
        system_prompt = prompts.get(PromptType.SYSTEM)
        if system_prompt == None:
            raise ValueError(
                f"System Prompt for {self.agent_type} not found!!"
            )  # sanity check

        return LLM(
            self.llm_provider,
            prompt_template=(
                system_prompt.text
                if system_prompt
                else f"you are a {self.agent_type} agent "
                + " who has complete understading of repo. With the given history of chat: {history} \nAnswer the following query with given info: {query}"
            ),
        )

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        # classify the query into agent needed or not
        classification_response = await self.classifier.run(ctx)
        classification = "AGENT_REQUIRED"
        logger.info(f"Classification response: {classification_response.response}")
        try:
            classification_json = json.loads(classification_response.response)
            if (
                classification_json
                and classification_json["classification"] == "LLM_SUFFICIENT"
            ):
                classification = "LLM_SUFFICIENT"
        except Exception as e:
            # use agent by default if classification failed
            logger.error(f"Defaulting to agent because classification failed: {e}")

        if classification == "AGENT_REQUIRED":
            return await self.rag_agent.run(ctx)

        # build llm response
        llm = await self._create_llm()
        return await llm.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        classification_response = await self.classifier.run(ctx)
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
            async for chunk in self.rag_agent.run_stream(ctx):
                yield chunk

            # You can pass the result to llm to stream response, but it's unnecessary and gives a overhead
            # rag_agent_response = await self.rag_agent.run(ctx)
            # ctx.query += f"\n with tool_response: {rag_agent_response.response} and citations: {rag_agent_response.citations}"

        # build llm response
        llm = await self._create_llm()
        async for chunk in llm.run_stream(ctx):
            yield chunk
