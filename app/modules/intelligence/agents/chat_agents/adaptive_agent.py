from typing import AsyncGenerator
from langchain_core.output_parsers import PydanticOutputParser
from app.modules.intelligence.agents.chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.prompts.classification_prompts import (
    ClassificationPrompts,
    ClassificationResponse,
    AgentType,
    ClassificationResult,
)
from app.modules.intelligence.prompts.prompt_service import PromptService, PromptType
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
        self.llm_provider = llm_provider
        self.prompt_provider = prompt_provider
        self.agent_type = agent_type
        self.rag_agent = rag_agent

    async def _get_messages(self, ctx: ChatContext):
        llm_prompts = await self.prompt_provider.get_prompts_by_agent_id_and_types(
            self.agent_type.value, [PromptType.SYSTEM, PromptType.HUMAN]
        )
        prompts = {prompt.type: prompt for prompt in llm_prompts}
        system_prompt = prompts.get(PromptType.SYSTEM)
        human_message_template = prompts.get(PromptType.HUMAN)

        if system_prompt == None:
            raise ValueError(
                f"System Prompt for {self.agent_type} not found!!"
            )  # sanity check

        query = ctx.query
        if human_message_template is not None:
            query = human_message_template.text.format(input=ctx.query)

        messages = [
            {"role": "system", "content": system_prompt.text},
            *[
                {
                    "role": "assistant",
                    "content": msg,
                }
                for msg in ctx.history
            ],
            {
                "role": "user",
                "content": query,
            },
        ]

        return messages

    async def _run_classification(self, ctx: ChatContext):
        inputs = {
            "query": ctx.query,
            "history": [msg for msg in ctx.history],
        }
        classification_prompt = ClassificationPrompts.get_classification_prompt(
            self.agent_type
        )

        parser = PydanticOutputParser(pydantic_object=ClassificationResponse)
        format_instructions = parser.get_format_instructions()

        messages = [
            {"role": "system", "content": classification_prompt},
            {
                "role": "user",
                "content": f"Query: {inputs['query']}\nHistory: {inputs['history']}\n\n{format_instructions}",
            },
        ]

        try:
            response = await self.llm_provider.call_llm(
                messages=messages, config_type="chat"
            )
            return parser.parse(response).classification  # type: ignore
        except Exception as e:
            logger.warning("Classification failed: %s", e)
            return ClassificationResult.AGENT_REQUIRED

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        # classify the query into agent needed or not
        classification = await self._run_classification(ctx)

        if classification == ClassificationResult.AGENT_REQUIRED:
            return await self.rag_agent.run(ctx)

        # build llm response
        messages = await self._get_messages(ctx)
        res = await self.llm_provider.call_llm(messages=messages, config_type="chat")
        return ChatAgentResponse(response=res, citations=[])  # type: ignore

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        classification = await self._run_classification(ctx)

        if classification == ClassificationResult.AGENT_REQUIRED:
            async for chunk in self.rag_agent.run_stream(ctx):
                yield chunk
            return
            # You can pass the result to llm to stream response, but it's unnecessary and gives a overhead
            # rag_agent_response = await self.rag_agent.run(ctx)
            # ctx.query += f"\n with tool_response: {rag_agent_response.response} and citations: {rag_agent_response.citations}"

        # build llm response
        messages = await self._get_messages(ctx)
        async for chunk in await self.llm_provider.call_llm(messages=messages, stream=True, config_type="chat"):  # type: ignore
            yield ChatAgentResponse(response=chunk, citations=[], tool_calls=[])
