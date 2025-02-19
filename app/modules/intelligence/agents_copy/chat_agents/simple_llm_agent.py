from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from .crewai_rag_agent import AgentType
from ..chat_agent import ChatAgent, ChatAgentResponse
from typing import List, Optional, AsyncGenerator
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)


class SimpleLLMAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, prompt_template: str):
        self.llm_large = llm_provider.get_large_llm(AgentType.LANGCHAIN)
        self.llm_small = llm_provider.get_small_llm(AgentType.LANGCHAIN)
        self.chain = self._create_chain(prompt_template)

    def _create_chain(self, prompt_template: str) -> RunnableSequence:
        parser = PydanticOutputParser(pydantic_object=ChatAgentResponse)
        prompt_with_parser = ChatPromptTemplate.from_template(
            template=prompt_template + "output format: {format_instructions}",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt_with_parser | self.llm_small | parser

    async def run(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> ChatAgentResponse:
        return await self.chain.ainvoke({"query": query, "history": history})

    async def run_stream(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self.chain.astream({"query": query, "history": history}):
            yield chunk
