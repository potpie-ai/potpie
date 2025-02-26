from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import CrewAIAgent, AgentConfig, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from ..langchain_agent import LangchainRagAgent
from typing import AsyncGenerator
import asyncio


class LowLevelDesignAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        return CrewAIAgent(
            self.llm_provider,
            AgentConfig(
                role="Design Planner",
                goal="Create a detailed low-level design plan for implementing new features",
                backstory="""
                    You are a senior software architect specializing in creating detailed,
                    actionable design plans. Your expertise lies in breaking down complex features into
                    manageable steps and providing clear guidance for implementation.      
                """,
                tasks=[
                    TaskConfig(
                        description=lld_task_prompt,
                        expected_output="Low-level design plan for implementing the new feature",
                    )
                ],
            ),
            tools=self.tools_provider.get_tools(
                [
                    "get_code_from_multiple_node_ids",
                    "get_node_neighbours_from_node_id",
                    "get_code_from_probable_node_name",
                    "ask_knowledge_graph_queries",
                    "get_nodes_from_tags",
                    "webpage_extractor",
                    "github_tool",
                ]
            ),
        )

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code Graph context of the node_ids in query:\n {code_results}"
            )

        file_structure = (
            await self.tools_provider.file_structure_tool.fetch_repo_structure(
                ctx.project_id
            )
        )
        ctx.additional_context += f"File Structure of the project:\n {file_structure}"

        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(await self._enriched_context(ctx))

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


lld_task_prompt = """
    Create a detailed low-level design plan for implementing the new feature. Your plan should include:
    1. A high-level overview of the implementation approach.
    2. Detailed steps for implementing the feature, including:
        - Specific files that need to be modified or created.
        - Proposed code changes or additions for each file.
        - Any new classes, methods, or functions that need to be implemented.
    3. Potential challenges or considerations for the implementation.
    4. Any suggestions for maintaining code consistency with the existing codebase.

    Use the provided tools to query the knowledge graph and retrieve or propose code snippets as needed.
    You can use the probable node name tool to get the code for a node by providing a partial file or function name.
"""
