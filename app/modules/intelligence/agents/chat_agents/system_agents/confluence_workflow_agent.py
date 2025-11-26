from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class ConfluenceWorkflowAgent(ChatAgent):
    """
    Specialized workflow agent for Confluence operations.
    This agent is designed to work within workflows and does not require repo context.
    It has access only to Confluence-specific tools.
    """

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
            role="Confluence Documentation Agent",
            goal="Create and manage Confluence documentation pages",
            backstory="""
                You are a specialized agent focused on creating high-quality documentation in Confluence.
                Your expertise includes:
                1. Creating well-structured Confluence pages with proper XHTML formatting
                2. Organizing content hierarchically within Confluence spaces
                3. Updating existing documentation with new information
                4. Adding contextual comments to pages
                
                You work within automated workflows to help teams maintain up-to-date documentation.
            """,
            tasks=[
                TaskConfig(
                    description=confluence_workflow_agent_prompt,
                    expected_output="Successfully created or updated Confluence page with proper formatting",
                )
            ],
        )

        # Only Confluence tools - no repo access needed
        tools = self.tools_provider.get_tools(
            [
                "list_confluence_integrations",
                "get_confluence_spaces",
                "get_confluence_page",
                "search_confluence_pages",
                "get_confluence_space_pages",
                "create_confluence_page",
                "update_confluence_page",
                "add_confluence_comment",
            ]
        )

        # Use PydanticRagAgent for simple, focused execution
        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


confluence_workflow_agent_prompt = """
    You are working within an automated workflow to create or update Confluence documentation.
    
    Your task:
    1. Analyze the input content and determine the best way to structure it for Confluence
    2. Use the appropriate Confluence tools to create or update pages
    3. Ensure proper XHTML formatting for Confluence compatibility
    4. Organize content with proper headings, lists, and formatting
    5. Add relevant metadata and parent page relationships
    
    Guidelines:
    - Use valid Confluence XHTML format (not Markdown)
    - Structure content with proper headings (h1, h2, h3)
    - Use lists, tables, and code blocks where appropriate
    - Keep formatting clean and professional
    - Ensure all required fields are provided (space key, title, content)
    
    If you encounter any issues or need clarification, provide clear error messages.
    Always confirm successful page creation or updates with relevant details (page ID, URL if available).
"""
