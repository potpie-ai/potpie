from app.modules.intelligence.agents.chat_agents.adaptive_agent import AdaptiveAgent
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.classification_prompts import AgentType
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator


class GeneralPurposeAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Potpie coding agent",
            goal="Answer user queries",
            backstory="""
                    You are a highly efficient and intelligent agent capable of generating precise and comprehensive responses.
                    Your tasks include:
                    1. Analyzing the user's query and formulating an effective strategy to help the user
                    2. Use tools and history to gather information regarding the query
                    3. Refining and enriching the initial results to provide a detailed and contextually appropriate response.
                    4. Maintaining traceability by including relevant citations and references in your output.
                    5. Including relevant citations in the response.
                """,
            tasks=[
                TaskConfig(
                    description=general_purpose_agent_prompt,
                    expected_output="Markdown formatted chat response to user's query grounded in provided context and tool results",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "webpage_extractor",
                "github_tool",
                "get_linear_issue",
                "update_linear_issue",
                "web_search_tool",
            ]
        )

        if self.llm_provider.is_current_model_supported_by_pydanticai(
            config_type="chat"
        ):
            return PydanticRagAgent(self.llm_provider, agent_config, tools)
        else:
            return AdaptiveAgent(
                llm_provider=self.llm_provider,
                prompt_provider=self.prompt_provider,
                rag_agent=CrewAIAgent(self.llm_provider, agent_config, tools),
                agent_type=AgentType.GENERAL,
            )

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


general_purpose_agent_prompt = """
    1. Compose response:
        - Organize results logically
        - Include citations and references
        - Provide comprehensive, focused answer

    2. Final review:
        - Check coherence and relevance
        - Identify areas for improvement


    Note:

    - Use markdown for code snippets with language name in the code block like python or javascript

    Ground your responses in provided code context and tool results. Use markdown for code snippets. Be concise and avoid repetition. If unsure, state it clearly.
    For debugging, unit testing, or unrelated code explanations, suggest specialized agents.
    Tailor your response based on question type:

    - New questions: Provide comprehensive answers
    - Follow-ups: Build on previous explanations from the chat history
    - Clarifications: Offer clear, concise explanations
    - Comments/feedback: Incorporate into your understanding

    Indicate when more information is needed. Use specific code references. Adapt to user's expertise level. Maintain a conversational tone and context from previous exchanges.
    Ask clarifying questions if needed. Offer follow-up suggestions to guide the conversation.
    Provide a comprehensive response with deep context, relevant file paths, include relevant code snippets wherever possible. Format it in markdown format.
"""
